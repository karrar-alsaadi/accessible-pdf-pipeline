import bisect
import csv
import itertools
import os
import re
import sys

import cv2
import numpy as np
import pymupdf
from rapidocr_onnxruntime import RapidOCR
from spellchecker import SpellChecker

# ===================== SETTINGS =====================
INPUT_FILE   = r"Pharmacology_1_2026_Part2_Adverse_Effects_of_Drugs_p20-30.pdf"
OUTPUT_FILE  = None          # None -> <input>_markers.pdf
KEEP_TMP     = False         # keep the intermediate *_ocr.pdf?

# Page markers. Words + full stop on purpose: bare digits glue together across
# pages ("9" + "10" -> "910") and the reader jumps to the wrong page.
START_T = "{n}."            # read FIRST on every page  -> "Page 10."
END_T   = "{n}."     # read LAST  on every page  -> "End of page 10."
MARKERS_VISIBLE = True       # True -> black visible text, False -> invisible (screen reader only)
MARKER_COLOR    = (0, 0, 0)  # RGB 0-1: (0,0,0)=black, (1,0,0)=red, (0,0,1)=blue

MIN_TEXT_CHARS   = 25        # page with fewer real chars -> treated as a full image page
SKIP_HEADER_Y    = 95        # images that end above this y (pt) are header logos -> skipped
MIN_IMG_PX       = 100       # ignore images smaller than this (px)
MIN_IMG_PT       = (80, 30)  # ignore images displayed smaller than this (width, height in pt)
LOW_CONF         = 0.80      # below this -> listed for review
AUTO_FIX_LOOKALIKES = True   # fix ONLY l/i, e/c letter mix-ups when the result is a real word
REBUILD_DPI      = 200       # image quality for rebuilt (glitch) pages
PAGE_OCR_DPI     = 300       # used when a whole page is an image

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
    r"C:\Windows\Fonts\segoeui.ttf",                     # Windows
    r"C:\Windows\Fonts\arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",                  # macOS
]
# ====================================================

FONT_FILE = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
if FONT_FILE is None:
    raise SystemExit("No Unicode font found. Install DejaVu: sudo apt install fonts-dejavu-core")
FONT = pymupdf.Font(fontfile=FONT_FILE)

# What the reader should say for symbols (applied to text added by this script)
SPOKEN = {
    "β": " beta ", "α": " alpha ", "γ": " gamma ", "δ": " delta ",
    "μ": " mu ", "µ": " mu ", "κ": " kappa ", "ω": " omega ",
    "→": " leads to ", "⇒": " leads to ", "↑": " increased ", "↓": " decreased ",
    "²⁺": " 2 plus ", "⁺": " plus ", "⁻": " minus ",
    "±": " plus or minus ", "≥": " greater than or equal to ",
    "≤": " less than or equal to ", "×": " times ", "°": " degrees ",
}

# Symbol-font glyphs that extract badly from Word/PDF text layers
TOKEN_FIX = {"à": "→"}                                   # Wingdings arrow seen as "à"
PUA_FIX = {"\uf0e0": "→", "\uf0ae": "→", "\uf0b7": "•", "\uf0a7": "•"}

_engine = None
_known = set()
review = []        # (page, text, reason)               -> _needs_review.txt
auto_fixed = []    # (page, before, after, reason)      -> _changes_log.txt
CORRECTIONS = {}   # (page, wrong_text) -> right_text   (from the CSV)
applied = []       # (page, wrong, right)
table_info = []    # (page, n_lines)


def engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


# ---------------------------------------------------------------- corrections CSV
def load_corrections(path):
    """CSV lines: page,wrong,right  (exact match of one OCR text box, that page only)."""
    c = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) == 3 and row[0].strip().isdigit():
                    c[(int(row[0]), row[1])] = row[2]
    return c


# ---------------------------------------------------------------- vocabulary / spelling
def build_vocab(doc):
    """English dictionary + every word in the document's own real text layer."""
    global _known
    sc = SpellChecker()
    _known = set(sc.word_frequency.dictionary.keys())
    for page in doc:
        for t in re.findall(r"[A-Za-z]{3,}", page.get_text("text")):
            _known.add(t.lower())


LOOK = {"l": "i", "i": "l", "e": "c", "c": "e"}


def _variants(tok):
    pos = [i for i, ch in enumerate(tok) if ch in LOOK]
    if not pos or len(pos) > 10:
        return []
    found = set()
    for r in range(1, min(len(pos), 5) + 1):
        for combo in itertools.combinations(pos, r):
            chars = list(tok)
            for i in combo:
                chars[i] = LOOK[chars[i]]
            v = "".join(chars)
            if v.lower() in _known:
                found.add(v)
    return sorted(found)


def check_text(text, page_no, conf):
    """Return the (possibly fixed) text. Flags doubtful words; never deletes anything."""
    if (page_no, text) in CORRECTIONS:                       # your reviewed CSV fix
        right = CORRECTIONS[(page_no, text)]
        applied.append((page_no, text, right))
        return right

    if AUTO_FIX_LOOKALIKES:
        # spacing / bullet only:  "13thweek" -> "13th week",  "OLect. = Lecture" -> "Lect. = Lecture"
        t2 = re.sub(r"(\d(?:st|nd|rd|th))(week)\b", r"\1 \2", text)
        t2 = re.sub(r"^O(?=[A-Z][a-z]+\.?\s*[=/])", "", t2)
        # roman numerals read as Il / Ill after a capitalised word:  "Biochemistry Il"
        t2 = re.sub(r"([A-Z][a-z]{3,}) ([Il1]{2,3})\b",
                    lambda m: m.group(1) + " " + "I" * len(m.group(2)), t2)
        if t2 != text:
            auto_fixed.append((page_no, text, t2, "spacing / bullet / roman numeral"))
            text = t2

    parts = re.split(r"([A-Za-z]+)", text)
    out = []
    for k, part in enumerate(parts):
        if part.isalpha() and len(part) >= 5 and part.lower() not in _known and not part.isupper():
            nxt = parts[k + 1] if k + 1 < len(parts) else ""
            if nxt[:1] in (".", "/"):                        # abbreviations: Pract./Tut.
                v = _variants(part)
                if len(v) == 1 and AUTO_FIX_LOOKALIKES:
                    auto_fixed.append((page_no, part, v[0], "l/i e/c look-alike"))
                    part = v[0]
                out.append(part)                             # never flagged (Pract, Lect...)
                continue
            # word + roman numeral glued together:  organieIII  Pharmacologyl
            stem = rn = None
            m = re.match(r"^([A-Za-z]{4,}?)(I{1,3})$", part)
            if m:
                stem, rn = m.group(1), m.group(2)
            elif part.endswith("yl") and len(part) >= 7 and part[:-1].lower() in _known:
                stem, rn = part[:-1], "I"
            if stem is not None:
                fixed = stem if stem.lower() in _known else None
                if fixed is None:
                    v = _variants(stem)
                    fixed = v[0] if len(v) == 1 else None
                if fixed is not None and AUTO_FIX_LOOKALIKES:
                    new = fixed + " " + rn
                    auto_fixed.append((page_no, part, new, "word + roman numeral"))
                    out.append(new)
                    continue
            v = _variants(part)
            if len(v) == 1 and AUTO_FIX_LOOKALIKES:
                auto_fixed.append((page_no, part, v[0], "l/i e/c look-alike"))
                part = v[0]
            else:
                hint = f" (maybe: {', '.join(v)})" if v else ""
                review.append((page_no, part, "unknown word" + hint))
        out.append(part)
    text = "".join(out)

    if conf < LOW_CONF:
        review.append((page_no, text, f"low confidence {conf:.2f}"))
    return text


# ---------------------------------------------------------------- hidden text writer
def speak(text):
    for sym, name in SPOKEN.items():
        text = text.replace(sym, name)
    return " ".join(text.split())


def write_hidden(page, lines):
    """Insert invisible text (render_mode=3), line by line, in the given order.
    Each line is shrunk/wrapped so it stays fully inside the page (text outside
    the page is lost by text extractors)."""
    total = 0
    W, H = page.rect.width, page.rect.height
    for text, x0, y0, x1, y1 in lines:
        text = speak(text)
        if not text:
            continue
        x0 = min(max(x0, 2), W - 20)
        fs = max(4.0, min((y1 - y0) * 0.8, 11.0))
        avail = W - x0 - 4
        chunks, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if FONT.text_length(trial, fs) <= avail or not cur:
                cur = trial
            else:
                chunks.append(cur)
                cur = word
        if cur:
            chunks.append(cur)
        y = min(max(y1 - (y1 - y0) * 0.2, fs), H - 3)
        for ch in chunks:
            page.insert_text((x0, min(y, H - 3)), ch, fontsize=fs,
                             fontname="dejavu", fontfile=FONT_FILE,
                             render_mode=3, overlay=True)
            y += fs + 1
            total += len(ch)
    return total


# ---------------------------------------------------------------- OCR + table structure
def _ocr_items(img):
    res, _ = engine()(img)
    items = []
    for box, txt, conf in (res or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append(dict(t=txt.strip(), x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys),
                          cx=(min(xs) + max(xs)) / 2, cy=(min(ys) + max(ys)) / 2,
                          h=max(ys) - min(ys), conf=float(conf)))
    return [i for i in items if i["t"]]


def _cluster_rows(items):
    items = sorted(items, key=lambda i: i["cy"])
    rows = []
    for it in items:
        for r in rows:
            if abs(it["cy"] - r["cy"]) < 0.6 * r["h"]:
                r["items"].append(it)
                n = len(r["items"])
                r["cy"] = (r["cy"] * (n - 1) + it["cy"]) / n
                break
        else:
            rows.append(dict(cy=it["cy"], h=it["h"], items=[it]))
    for r in rows:
        r["items"].sort(key=lambda i: i["x0"])
    return rows


def _union(items):
    return (min(i["x0"] for i in items), min(i["y0"] for i in items),
            max(i["x1"] for i in items), max(i["y1"] for i in items))


def _merge(vals, tol):
    vals = sorted(vals)
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def _grid_lines(img, scale, text_h):
    """Find long horizontal / vertical ruling lines (table borders)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (22 * scale, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(max(18 * scale, 1.2 * text_h))))
    hmask = cv2.morphologyEx(b, cv2.MORPH_OPEN, hk)
    vmask = cv2.morphologyEx(b, cv2.MORPH_OPEN, vk)
    hl, vl = [], []
    n, _, st, _ = cv2.connectedComponentsWithStats(hmask)
    for k in range(1, n):
        x, y, w, h, _a = st[k]
        if h <= 8 * scale:
            hl.append((x, y + h / 2, x + w))
    n, _, st, _ = cv2.connectedComponentsWithStats(vmask)
    for k in range(1, n):
        x, y, w, h, _a = st[k]
        if w <= 8 * scale:
            vl.append((x + w / 2, y, y + h))
    return hl, vl


def _recover(img, b0, b1, xa, xb, scale):
    """Re-read a table cell that came out empty although it contains ink (e.g. a tiny digit)."""
    m = 4 * scale
    y0, y1, x0, x1 = int(b0 + m), int(b1 - m), int(xa + m), int(xb - m)
    if y1 - y0 < 12 or x1 - x0 < 8:
        return None
    crop = img[y0:y1, x0:x1]
    if (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 140).mean() < 0.004:
        return None                                       # really empty
    pad = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    pad = cv2.resize(pad, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    its = sorted(_ocr_items(pad), key=lambda i: (i["cy"], i["x0"]))
    return " ".join(i["t"] for i in its) or None


def _structure(img, items, hl, vl, H, W, scale, page_no):
    """Turn OCR boxes into reading-order lines. Table rows become one line per row,
    cells separated by ' ; '. Empty cells that contain ink are re-read; rows with an
    unusual number of cells are flagged for review."""
    tol = 5 * scale
    ys = _merge([y for _, y, _ in hl], tol)
    bounds = [0.0] + ys + [float(H)]
    lines, table_rows = [], []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        if b1 - b0 < tol:
            continue
        band = [i for i in items if b0 <= i["cy"] < b1]
        cols = _merge([x for x, y0, y1 in vl
                       if min(y1, b1) - max(y0, b0) >= 0.6 * (b1 - b0)], tol)
        if not band and not cols:
            continue
        if cols:
            cells = {}
            for it in band:
                cells.setdefault(bisect.bisect(cols, it["cx"]), []).append(it)
            if len(cells) >= 3:                           # a real table row: recover missed digits
                lo, hi = min(cells), max(cells)
                for idx in range(lo + 1, hi):
                    if idx in cells:
                        continue
                    txt = _recover(img, b0, b1, cols[idx - 1], cols[idx], scale)
                    if txt and re.fullmatch(r"\d{1,4}%?", txt.strip()):
                        txt = txt.strip()
                        review.append((page_no, txt, "number recovered from a cell the first read missed"))
                        cells[idx] = [dict(t=txt, x0=cols[idx - 1], x1=cols[idx], y0=b0, y1=b1,
                                           cx=(cols[idx - 1] + cols[idx]) / 2, cy=(b0 + b1) / 2,
                                           h=b1 - b0, conf=1.0)]
            if not cells:
                continue
            parts = []
            for idx in sorted(cells):
                parts.append(" ".join(" ".join(i["t"] for i in r["items"])
                                      for r in _cluster_rows(cells[idx])))
            allit = [i for v in cells.values() for i in v]
            table_rows.append((len(lines), len(parts)))
            lines.append((" ; ".join(parts), *_union(allit), True))
        else:
            for r in _cluster_rows(band):
                lines.append((" ".join(i["t"] for i in r["items"]), *_union(r["items"]), False))
    # rows whose cell count differs from the usual one (a cell may have been missed)
    counts = [n for _, n in table_rows if n >= 5]
    if counts:
        usual = max(set(counts), key=counts.count)
        for li, n in table_rows:
            if abs(n - usual) == 1 and usual >= 5:
                review.append((page_no, lines[li][0][:60], f"row has {n} cells, usual is {usual}"))
    return lines


def ocr_image_to_lines(img, bbox, page_no):
    """OCR one image placed at bbox (page points). Returns hidden-text lines in page coords."""
    h0, w0 = img.shape[:2]
    scale = 3 if max(h0, w0) < 1200 else 2
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    items = _ocr_items(img)
    for it in items:                                     # spell / correction pass per box
        it["t"] = check_text(it["t"], page_no, it["conf"])
    text_h = float(np.median([i['h'] for i in items])) if items else 0.0
    hl, vl = _grid_lines(img, scale, text_h)
    lines = _structure(img, items, hl, vl, H, W, scale, page_no)
    sx, sy = bbox.width / W, bbox.height / H
    out = []
    n_table = 0
    for text, x0, y0, x1, y1, is_table in lines:
        n_table += is_table
        out.append((text, bbox.x0 + x0 * sx, bbox.y0 + y0 * sy,
                    bbox.x0 + x1 * sx, bbox.y0 + y1 * sy))
    if n_table:
        table_info.append((page_no, n_table))
    return out


def load_image(doc, page, info):
    """Native image if possible (best quality), else a render of its area."""
    bbox = pymupdf.Rect(info["bbox"]) & page.rect
    try:
        if info.get("xref", 0) <= 0:
            raise ValueError("inline image")
        pix = pymupdf.Pixmap(doc, info["xref"])
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        if pix.colorspace is None or pix.colorspace.n != 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    except Exception:
        pix = page.get_pixmap(dpi=PAGE_OCR_DPI, clip=bbox)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    return img, bbox


def usable_images(page):
    out = []
    for info in page.get_image_info(xrefs=True):
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if r.is_empty or info["width"] < MIN_IMG_PX or info["height"] < MIN_IMG_PX:
            continue
        if r.width < MIN_IMG_PT[0] or r.height < MIN_IMG_PT[1]:
            continue
        if r.y1 < SKIP_HEADER_Y:                         # header logos
            continue
        out.append(info)
    return out


def page_image_lines(doc, page):
    """OCR every real image on the page (tables, diagrams, scans)."""
    pn = page.number + 1
    lines = []
    imgs = usable_images(page)
    if imgs:
        for info in imgs:
            img, bbox = load_image(doc, page, info)
            lines += ocr_image_to_lines(img, bbox, pn)
    elif len(page.get_text("text").strip()) < MIN_TEXT_CHARS:      # whole page is a picture
        pix = page.get_pixmap(dpi=PAGE_OCR_DPI)
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        lines += ocr_image_to_lines(img, page.rect, pn)
    return lines


def ocr_stage(doc, skip_pages):
    report = {}
    for n, page in enumerate(doc, start=1):
        if n in skip_pages:
            continue
        lines = page_image_lines(doc, page)
        if lines:
            report[n] = write_hidden(page, lines)
    return report


# ---------------------------------------------------------------- glitch pages
def fix_token(tok):
    if tok in TOKEN_FIX:
        return TOKEN_FIX[tok]
    out = []
    for c in tok:
        if c in PUA_FIX:
            out.append(PUA_FIX[c])
        elif "\uf000" <= c <= "\uf0ff":
            continue                       # unknown private-use glyph: drop
        else:
            out.append(c)
    return "".join(out)


def page_has_glitch(page):
    for w in page.get_text("words"):
        t = w[4]
        if t in TOKEN_FIX or any("\uf000" <= c <= "\uf0ff" for c in t):
            return True
    return False


def rebuild_page(out_doc, src_doc, src_page):
    """Page with symbol-font glitches: keep the look (as image) and put a corrected
    invisible text layer on top (page text first, then text found in its images)."""
    page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    page.insert_image(page.rect, pixmap=src_page.get_pixmap(dpi=REBUILD_DPI))
    lines = []
    for b in src_page.get_text("dict", sort=True)["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            raw = "".join(sp["text"] for sp in l["spans"])
            fixed = " ".join(fix_token(t) for t in raw.split())
            if fixed:
                x0, y0, x1, y1 = l["bbox"]
                lines.append((fixed, x0, y0, x1, y1))
    lines += page_image_lines(src_doc, src_page)         # figures on the same page
    write_hidden(page, lines)


def fix_stage(src):
    out = pymupdf.open()
    rebuilt = set()
    for i, page in enumerate(src):
        if page_has_glitch(page):
            rebuild_page(out, src, page)
            rebuilt.add(i + 1)
        else:
            out.insert_pdf(src, from_page=i, to_page=i)
    return out, rebuilt


# ---------------------------------------------------------------- markers
def add_page_markers(input_pdf, output_pdf, start_template=START_T,
                     end_template=END_T, fontsize=10, margin=12, visible=None):
    """START marker = first thing in the page's content stream, END marker = last.
    Text ends with '.' + a space so it can never glue to a neighbouring page."""
    if visible is None:
        visible = MARKERS_VISIBLE
    mode = 0 if visible else 3                # 0 = normal (visible), 3 = invisible
    doc = pymupdf.open(input_pdf)
    for n, page in enumerate(doc, start=1):
        page.clean_contents()                 # normalise streams before inserting
        r = page.rect
        kw = dict(fontsize=fontsize, fontname="helv", render_mode=mode, color=MARKER_COLOR)
        page.insert_text((margin, margin + fontsize),
                         start_template.format(n=n) + " ", overlay=False, **kw)
        page.insert_text((margin, r.height - margin),
                         end_template.format(n=n) + " ", overlay=True, **kw)
    doc.save(output_pdf, garbage=3, deflate=True)
    doc.close()


def _extractors(pdf):
    """Independent ways a reader may pull text out of the file."""
    d = pymupdf.open(pdf)
    yield "pymupdf-stream", [" ".join(w[4] for w in p.get_text("words", sort=False)) for p in d]
    yield "pymupdf-sorted", [" ".join(w[4] for w in p.get_text("words", sort=True)) for p in d]
    yield "pymupdf-text",   [p.get_text() for p in d]
    try:
        from pypdf import PdfReader          # optional: pip install pypdf
        yield "pypdf", [p.extract_text() or "" for p in PdfReader(pdf).pages]
    except ImportError:
        pass


def verify(pdf, start_template=START_T, end_template=END_T):
    """Strict check. Returns a list of problems (empty list = everything OK).
    The pages are glued together with '', newline and space (like different readers do);
    every marker must be unique, in order, and nothing may sit between pages."""
    bad = []
    for name, pages in _extractors(pdf):
        for sep in ("", "\n", " "):
            whole = sep.join(pages)
            pos = 0
            for n in range(1, len(pages) + 1):
                s, e = start_template.format(n=n), end_template.format(n=n)
                if len(re.findall(re.escape(s), whole)) != 1:
                    bad.append((name, n, "start marker not unique"))
                if len(re.findall(re.escape(e), whole)) != 1:
                    bad.append((name, n, "end marker not unique"))
                i, j = whole.find(s, pos), whole.find(e, pos)
                if i < 0 or j < i:
                    bad.append((name, n, "order broken"))
                    continue
                if whole[pos:i].strip():
                    bad.append((name, n, "stray text before start marker"))
                span = whole[i + len(s): j]
                if re.search(r"(?<![A-Za-z])Page \d+\.|End of page \d+\.", span):
                    bad.append((name, n, "foreign marker inside page"))
                pos = j + len(e)
            if whole[pos:].strip():
                bad.append((name, "end", "text after last marker"))
    return sorted(set(bad), key=str)


# ---------------------------------------------------------------- reports
def dump_reading_text(pdf, txt_path):
    doc = pymupdf.open(pdf)
    with open(txt_path, "w", encoding="utf-8") as f:
        for n, p in enumerate(doc, start=1):
            words = p.get_text("words", sort=False)
            f.write(f"\n===== PAGE {n} =====\n" + " ".join(w[4] for w in words) + "\n")
    doc.close()


def dump_review(path, rebuilt, ocr_report):
    with open(path, "w", encoding="utf-8") as f:
        f.write("PAGES WITH TEXT ADDED FROM IMAGES (chars): " + (str(ocr_report) if ocr_report else "none") + "\n")
        f.write("PAGES REBUILT (symbol glitches fixed): " + (str(sorted(rebuilt)) if rebuilt else "none") + "\n")
        f.write("TABLE ROWS DETECTED (page, rows): " + (str(table_info) if table_info else "none") + "\n\n")
        f.write("WORDS TO CHECK AGAINST THE PAGE IMAGE:\n")
        for page, text, why in review:
            f.write(f"  page {page:>3} | {why:<28} | {text}\n")
        if not review:
            f.write("  none\n")


def dump_changes(path):
    used = {(p, w) for p, w, _ in applied}
    missing = [k for k in CORRECTIONS if k not in used]
    with open(path, "w", encoding="utf-8") as f:
        f.write("AUTOMATIC FIXES (look-alike letters / roman numerals only):\n")
        for p, a, b, why in auto_fixed:
            f.write(f"  page {p} | {a!r} -> {b!r} | {why}\n")
        if not auto_fixed:
            f.write("  none\n")
        f.write("\nCSV CORRECTIONS APPLIED:\n")
        for p, w, r in applied:
            f.write(f"  page {p} | {w!r} -> {r!r}\n")
        if not applied:
            f.write("  none\n")
        f.write("\nCSV CORRECTIONS NOT MATCHED (check page number / exact spelling):\n")
        for p, w in missing:
            f.write(f"  page {p} | {w!r}\n")
        if not missing:
            f.write("  none\n")
    return len(auto_fixed), len(applied), len(missing)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    if len(sys.argv) > 1:                            # optional: python script.py file.pdf
        INPUT_FILE = sys.argv[1]
    base, ext = os.path.splitext(INPUT_FILE)
    tmp = base + "_ocr" + ext
    out_file = OUTPUT_FILE or (base + "_markers" + ext)

    CORRECTIONS.update(load_corrections(base + "_corrections.csv"))
    print("Font in use:", FONT_FILE)
    print("CSV corrections loaded:", len(CORRECTIONS))

    src = pymupdf.open(INPUT_FILE)
    build_vocab(src)                                 # dictionary + the PDF's own words
    doc, rebuilt = fix_stage(src)                    # 1) fix symbol-font glitches
    ocr_report = ocr_stage(doc, skip_pages=rebuilt)  # 2) OCR images, tables -> rows
    src.close()
    try:
        doc.subset_fonts()
    except Exception:
        pass
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()

    add_page_markers(tmp, out_file)                  # 3) markers LAST
    bad = verify(out_file)

    dump_reading_text(out_file, base + "_reading_check.txt")
    dump_review(base + "_needs_review.txt", rebuilt, ocr_report)
    n_auto, n_csv, n_missing = dump_changes(base + "_changes_log.txt")
    if not KEEP_TMP:
        os.remove(tmp)

    print("Saved:", out_file)
    print("Pages rebuilt (glitch fix):", sorted(rebuilt) or "none")
    print("Pages with text added from images (chars):", ocr_report or "none")
    print("Table rows detected (page, rows):", table_info or "none")
    print("Automatic look-alike fixes:", n_auto, "| CSV applied:", n_csv, "| CSV not matched:", n_missing)
    print("Words to check:", len(review))
    print("Marker check:", "ALL OK" if not bad else bad)







