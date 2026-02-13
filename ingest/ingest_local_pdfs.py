import os
import logging
import tomli
import chromadb
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_extractor import extract_text_from_pdf
from text_cleaner import clean_extracted_text

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
def load_configuration() -> Optional[Dict[str, Any]]:
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomli.load(f)
    except FileNotFoundError:
        logging.error("❌ .streamlit/secrets.toml not found")
        return None

    config = {
        "CHROMA_API_KEY": secrets.get("CHROMA_API_KEY"),
        "CHROMA_TENANT": secrets.get("CHROMA_TENANT"),
        "CHROMA_DATABASE": secrets.get("CHROMA_DATABASE"),
        "PDF_BASE_PATH": secrets.get("general", {}).get("pdf_base_path", "data/pdfs"),
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 150,
    }

    required = ["CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"]
    for key in required:
        if not config.get(key):
            logging.error(f"❌ Missing config value: {key}")
            return None

    return config


# --------------------------------------------------
# PROCESS PDFs
# --------------------------------------------------
def process_company_folders(
    pdf_base_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, List[Document]]:

    base_path = Path(pdf_base_path)

    if not base_path.exists():
        logging.error(f"❌ PDF base path not found: {base_path}")
        return {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    company_docs: Dict[str, List[Document]] = {}

    for company_dir in base_path.iterdir():
        if not company_dir.is_dir():
            continue

        company_name = company_dir.name
        collection_name = company_name.lower().replace(" ", "_")
        company_docs[collection_name] = []

        logging.info(f" Processing company: {company_name}")

        for pdf_file in company_dir.glob("*.pdf"):
            logging.info(f" Processing PDF: {pdf_file.name}")

            extract_text_from_pdf(str(pdf_file))
            extracted_txt = pdf_file.with_name(f"{pdf_file.stem}_extracted.txt")

            cleaned_text = clean_extracted_text(str(extracted_txt))
            if not cleaned_text:
                continue

            chunks = text_splitter.split_text(cleaned_text)

            for chunk in chunks:
                metadata = {
                    "company": company_name,
                    "document_name": pdf_file.stem,
                    "pdf_file": pdf_file.name,
                }

                company_docs[collection_name].append(
                    Document(page_content=chunk, metadata=metadata)
                )

            logging.info(f"✅ Created {len(chunks)} chunks from {pdf_file.name}")

    return company_docs


# --------------------------------------------------
# INGEST INTO CHROMA
# --------------------------------------------------
def ingest_into_chroma(company_docs: Dict[str, List[Document]], config: Dict[str, Any]) -> None:

    # 🔥 Local embeddings instead of Google
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.CloudClient(
        api_key=config["CHROMA_API_KEY"],
        tenant=config["CHROMA_TENANT"],
        database=config["CHROMA_DATABASE"]
    )

    for collection_name, docs in company_docs.items():
        if not docs:
            continue

        logging.info(f" Ingesting {len(docs)} chunks into '{collection_name}'")

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

        logging.info(f"✅ Ingestion complete for {collection_name}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    logging.info("==== LOCAL PDF INGESTION STARTED ====")

    config = load_configuration()
    if not config:
        return

    company_docs = process_company_folders(
        pdf_base_path=config["PDF_BASE_PATH"],
        chunk_size=config["CHUNK_SIZE"],
        chunk_overlap=config["CHUNK_OVERLAP"],
    )

    ingest_into_chroma(company_docs, config)

    logging.info("==== INGESTION FINISHED SUCCESSFULLY ====")


if __name__ == "__main__":
    main()

