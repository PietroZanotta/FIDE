import fitz  # PyMuPDF
from pathlib import Path

def pdf_to_png(pdf_path, output_dir="output_png", dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(pdf_path)

    for page_number, page in enumerate(pdf, start=1):
        pix = page.get_pixmap(dpi=dpi)
        output_file = output_dir / f"page_{page_number}.png"
        pix.save(output_file)
        print(f"Saved: {output_file}")

    pdf.close()


pdf_to_png("fide_diag3.pdf")