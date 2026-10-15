# Accessible PDF Pipeline for Blind Students

A small Python pipeline that prepares lecture PDFs for screen readers and reader apps.

It does two things:

1. **Makes scanned PDFs searchable.** Each page is OCR'd with Tesseract and an invisible text layer is placed over the original page image, so the text can be selected, copied, and read aloud.
2. **Adds page-number markers.** A visible `{n}`-style page number is placed at the very top and very bottom of each page, so a reader can say "read from page N" and the reader knows where each page starts and ends.

The whole process runs locally with free tools. No paid APIs are used.

## Requirements

- Python 3.9+
- [PyMuPDF](https://pymupdf.readthedocs.io/) (`pip install pymupdf`)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - Linux: `sudo apt install tesseract-ocr`
  - Windows: [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
  - For Arabic, also install the `ara` language data.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Usage

Open the script and set the input file and language at the bottom:

```python
INPUT_FILE = "your_file.pdf"   # path to the scanned or digital PDF
LANGUAGE   = "eng"             # "eng" for English, "ara" for Arabic
```

Then run:

```bash
python make_accessible_pdf.py
```

The script produces:

| File | Contents |
|---|---|
| `<name>_searchable.pdf` | Scanned pages with an invisible OCR text layer |
| `<name>_searchable_with_markers.pdf` | Final file with page markers, ready to send to the student |

At the end, the script re-reads every page and checks that the first and last lines of text match the page number.

## How the markers work

- The page image is re-inserted into a slightly smaller rectangle, leaving a 14 pt strip at the top and bottom of each page.
- The start marker and end marker are written as separate content streams and are placed in the page's `/Contents` array in reading order: start marker, original body, end marker.
- Markers are visible black text (not invisible), because reader apps skip invisible text.

## Known limitations

- **Table pages and image-only content** are not converted to text. Only OCR'd text is readable. Tables stored as images need separate handling.
- **Markers may be read aloud** by some reader apps as part of the page content. This is a known issue.
- **OCR quality** depends on scan quality. Review the output before sending it to a student.
- **Page layout** is preserved as an image. Reflowed text is not produced.

## Project structure

```
.
├── make_accessible_pdf.py   # main pipeline (OCR + markers + verification)
├── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
```

## License

Add a license of your choice (for example MIT) before publishing.
