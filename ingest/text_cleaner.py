import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clean_extracted_text(extracted_txt_path: str) -> Optional[str]:
    """
    Cleans extracted PDF text and saves a cleaned version as a new file
    with suffix '_cleaned.txt' in the same directory.

    Args:
        extracted_txt_path (str): Path to the *_extracted.txt file.

    Returns:
        str: Cleaned text if successful, else None.
    """

    extracted_txt_path = Path(extracted_txt_path)

    if not extracted_txt_path.exists():
        logging.error(f"Extracted text file not found: {extracted_txt_path}")
        return None

    try:
        logging.info(f"Cleaning extracted text: {extracted_txt_path}")

        with open(extracted_txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # -------- CLEANING STEPS --------

        # 1. Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Remove excessive spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # 3. Fix broken words across line breaks (e.g., finan-\ncial → financial)
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # 4. Merge lines that break mid-sentence
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # 5. Reduce multiple newlines to max two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 6. Remove isolated page numbers (common in PDFs)
        text = re.sub(r"\n\d+\n", "\n", text)

        # 7. Trim leading/trailing whitespace
        text = text.strip()

        if not text:
            logging.warning("Cleaned text is empty after processing.")
            return None

        # -------- SAVE CLEANED TEXT --------

        cleaned_file_path = extracted_txt_path.with_name(
            extracted_txt_path.stem.replace("_extracted", "_cleaned") + ".txt"
        )

        with open(cleaned_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        logging.info(f"Cleaned text saved to: {cleaned_file_path}")

        return text

    except Exception as e:
        logging.error(f"Error cleaning extracted text {extracted_txt_path}: {e}")
        return None
