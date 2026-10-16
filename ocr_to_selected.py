# """
# Complete accessibility pipeline for blind students.
#
# Step 1 — Convert scanned/OCR PDF → searchable PDF (selectable text)
# Step 2 — Embed slide markers INSIDE the body content stream:
#          • "slide: X, started" as the very first text in the body
#          • "slide: X, end"     as the very last  text in the body
#
# No separate header/footer streams — markers live inside the body itself,
# so the screen reader reads them in the natural order:
#     slide: 1, started → content → slide: 1, end
#
# Requirements:
#     pip install pymupdf --break-system-packages
#     sudo apt install tesseract-ocr -y        (Linux)
#     Windows: https://github.com/UB-Mannheim/tesseract/wiki
# """
#


import re
import os
import pymupdf as fitz  # هذه الطريقة أكثر أماناً في النسخ الحديثة

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Convert scanned/OCR PDF → searchable PDF
# ══════════════════════════════════════════════════════════════════

def make_searchable_pdf(input_pdf, output_pdf=None, lang="eng"):
    if output_pdf is None:
        base, ext = os.path.splitext(input_pdf)
        output_pdf = base + "_searchable" + ext

    print(f"\n[Step 1] Converting scanned PDF to searchable PDF...")
    src = fitz.open(input_pdf)
    new_doc = fitz.open()

    for i, src_page in enumerate(src):
        print(f"  Page {i + 1}/{len(src)} ...", end=" ", flush=True)

        w = src_page.rect.width
        h = src_page.rect.height

        # Run Tesseract OCR via PyMuPDF
        ocr_tp = src_page.get_textpage_ocr(language=lang, dpi=300, full=True)
        words = src_page.get_text("words", textpage=ocr_tp)

        new_page = new_doc.new_page(width=w, height=h)

        # Original page as sharp background image
        pix = src_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        new_page.insert_image(fitz.Rect(0, 0, w, h), stream=pix.tobytes("jpeg"))

        # Invisible text layer on top (render_mode=3 = invisible)
        inserted = 0
        for word_data in words:
            x0, y0, x1, y1, word = word_data[:5]
            if not word.strip():
                continue
            box_h = y1 - y0
            new_page.insert_text(
                (x0, y1),
                word + " ",
                fontsize=max(box_h, 8),
                fontname="helv",
                color=(0, 0, 0),
                render_mode=3,
                overlay=True,
            )
            inserted += 1
        print(f"{inserted} words")

    new_doc.save(output_pdf, garbage=4, deflate=True)
    new_doc.close()
    src.close()
    print(f"  Saved: {output_pdf}")
    return output_pdf



# ══════════════════════════════════════════════════════════════════
# STEP 2 — Embed markers by adding new content streams (no modification of existing streams)
# ══════════════════════════════════════════════════════════════════



def _make_end_marker_stream(doc, text, page_height):
    y = 8                         # near the BOTTOM of page (y=8 from bottom edge)
    stream_data = (
        b'q BT '
        b'/helv 8 Tf '
        b'0 0 0 rg '              # black fill color
        b'0 Tr '                  # render_mode 0 = FILL (VISIBLE)
        b'1 0 0 1 5 ' + str(y).encode() + b' Tm '
        b'(' + text.encode('latin-1', errors='replace') + b') Tj '
        b'ET Q\n'
    )
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, stream_data)
    return xref



def _make_marker_stream(doc, text, y_position):
    """
    Creates a visible black text marker stream at the given y position.
    render_mode=0 (fill) = fully visible black text.
    """
    stream_data = (
        b'q '
        b'BT '
        b'/helv 7 Tf '
        b'0 0 0 rg '                          # black color
        b'0 Tr '                               # render_mode 0 = visible fill
        b'1 0 0 1 5 ' + str(round(y_position)).encode() + b' Tm '
        b'(' + text.encode('latin-1', errors='replace') + b') Tj '
        b'ET '
        b'Q\n'
    )
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, stream_data)
    return xref


def embed_markers_in_body(input_pdf, output_pdf=None):
    if output_pdf is None:
        base, ext = os.path.splitext(input_pdf)
        output_pdf = base + "_with_markers" + ext

    MARKER_STRIP_H = 14          # height in points reserved for each marker strip

    print(f"\n[Step 2] Embedding slide markers inside body content...")
    doc = fitz.open(input_pdf)

    for page_num in range(len(doc)):
        page = doc[page_num]
        slide_num = page_num + 1
        page_w = page.rect.width
        page_h = page.rect.height

        # ── Re-insert background image into a SHRUNK rect ──────────────────
        # leaving a strip at top and bottom for our markers
        # First, grab the existing page pixmap BEFORE we modify anything
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        # Clear the page entirely, then re-draw background in shrunk rect
        page.clean_contents()

        # Image rect: leave MARKER_STRIP_H pts at top and bottom
        img_rect = fitz.Rect(
            0,
            MARKER_STRIP_H,          # top strip reserved for "started" marker
            page_w,
            page_h - MARKER_STRIP_H  # bottom strip reserved for "end" marker
        )
        page.insert_image(img_rect, stream=pix.tobytes("jpeg"), overlay=False)

        # ── Create marker streams ───────────────────────────────────────────
        # PDF y=0 is BOTTOM of page
        # "started" marker → top strip  → y near page_h
        # "end"     marker → bottom strip → y near 0

        start_y = page_h - (MARKER_STRIP_H - 3)   # e.g. page_h=792 → y=781
        end_y   = 4                                 # near the bottom edge

        start_xref = _make_marker_stream(doc, f" {slide_num} ", start_y) # slide: {slide_num}, started
        end_xref   = _make_marker_stream(doc, f" {slide_num} ",     end_y) # slide: {slide_num}, enddd

        # ── Rebuild /Contents in reading order: start → body → end ─────────
        existing = page.get_contents()   # after clean_contents + insert_image

        new_contents = [start_xref] + list(existing) + [end_xref]
        refs     = " ".join(f"{x} 0 R" for x in new_contents)
        array_str = f"[ {refs} ]"

        page_xref = page.xref
        page_dict = doc.xref_object(page_xref, compressed=False)
        page_dict = re.sub(r'/Contents\s+\d+\s+\d+\s+R', '', page_dict)
        page_dict = re.sub(r'/Contents\s*\[.*?\]', '', page_dict, flags=re.DOTALL)
        page_dict = page_dict.rstrip()
        if page_dict.endswith('>>'):
            page_dict = page_dict[:-2] + f'\n/Contents {array_str}\n>>'
        else:
            page_dict += f'\n/Contents {array_str}\n'

        doc.update_object(page_xref, page_dict)
        print(f"  Page {slide_num}: ✓")

    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    print(f"  Saved: {output_pdf}")
    return output_pdf

# ══════════════════════════════════════════════════════════════════
# VERIFY — Check reading order on all pages
# ══════════════════════════════════════════════════════════════════

def verify(pdf_path):
    print(f"\n[Verify] Checking reading order in {pdf_path}...")
    doc = fitz.open(pdf_path)
    all_ok = True
    for i in range(len(doc)):
        lines = [l.strip() for l in doc[i].get_text().split('\n') if l.strip()]
        first = lines[0]  if lines else ''
        last  = lines[-1] if lines else ''
        ok = (first == f'{i+1}') and (last == f'{i+1}')
        status = '✓' if ok else '✗'
        if not ok:
            print(f"  Page {i+1:2d}: {status}  first='{first[:50]}'  last='{last[:40]}'")
            all_ok = False
        else:
            print(f"  Page {i+1:2d}: {status}")
    doc.close()
    if all_ok:
        print(f"  All {i+1} pages correct ✓")
    return all_ok


# ══════════════════════════════════════════════════════════════════
# RUN THE FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    INPUT_FILE = "dd.pdf"  # ← your scanned PDF name here
    LANGUAGE   = "eng"             # "eng" = English | "ara" = Arabic

    # Step 1: scanned PDF → searchable PDF
    searchable = make_searchable_pdf(INPUT_FILE, lang=LANGUAGE)

    # Step 2: embed slide markers directly inside body streams
    final = embed_markers_in_body(searchable)

    # Verify
    verify(final)

    print(f"\n✓ Done!  Send this file to the student: {final}")