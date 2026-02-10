import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_text_from_pdf("E:\8th Sem\MAJOR\Documind\data\pdfs\01.pdf": str) -> Optional[str]:
    """
    Extracts text from a local PDF file and saves it as a .txt file
    with suffix '_extracted' in the same directory.

    Args:
        pdf_path (str): Path to the local PDF file.

    Returns:
        str: Extracted text if successful, else None.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logging.error(f"PDF file not found: {pdf_path}")
        return None

    try:
        logging.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        extracted_pages = []

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            page_text = page.get_text("text")

            if page_text and page_text.strip():
                extracted_pages.append(page_text)

        doc.close()

        if not extracted_pages:
            logging.warning(f"No readable text found in PDF: {pdf_path}")
            return None

        full_text = "\n\n".join(extracted_pages)

        # ---- Save extracted text to file ----
        extracted_file_path = pdf_path.with_name(
            f"{pdf_path.stem}_extracted.txt"
        )

        with open(extracted_file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        logging.info(f"Extracted text saved to: {extracted_file_path}")

        return full_text

    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None
