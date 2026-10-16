# OCR Accessibility Pipeline

Converts scanned PDFs into accessible, searchable PDFs for blind students, and embeds slide markers so screen readers announce each slide's start and end in reading order.

## Pipeline

1. **Make searchable** — runs Tesseract OCR on each page and adds an invisible text layer over the original page image.
2. **Embed slide markers** — adds `slide: N` markers at the top (start) and bottom (end) of every page's content stream.
3. **Verify** — checks that each page's first and last text lines match its slide number.

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - Linux: `sudo apt install tesseract-ocr`
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
- For Arabic OCR, install the `ara` language pack for Tesseract.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your scanned PDF in the project folder.
2. Edit the settings at the bottom of `ocr_to_selected.py`:
   ```python
   INPUT_FILE = "dd.pdf"
   LANGUAGE   = "eng"   # or "ara"
   ```
3. Run:
   ```bash
   python ocr_to_selected.py
   ```

Output files:
- `dd_searchable.pdf` — OCR text layer added
- `dd_searchable_with_markers.pdf` — final file to send to the student

## Notes

- Input PDFs and generated output are not tracked in git (see `.gitignore`).
