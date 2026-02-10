import os
import logging
import tomli
import chromadb
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pdf_extractor import extract_text_from_pdf
from text_cleaner import clean_extracted_text

# --------------------------------------------------
# 0. LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
def load_configuration() -> Optional[Dict[str, Any]]:
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomli.load(f)
    except FileNotFoundError:
        logging.error("❌ .streamlit/secrets.toml not found")
        return None

    config = {
        "GOOGLE_API_KEY": secrets.get("GEMINI_API_KEY"),
        "CHROMA_CLOUD_API_KEY": secrets.get("CHROMA_CLOUD_API_KEY"),
        "CHROMA_CLOUD_HOST": secrets.get("CHROMA_CLOUD_HOST"),
        "PDF_BASE_PATH": secrets.get("general", {}).get("pdf_base_path", "data/pdfs"),
        "EMBEDDING_MODEL": "models/text-embedding-004",
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 150,
    }

    required = ["GOOGLE_API_KEY", "CHROMA_CLOUD_API_KEY", "CHROMA_CLOUD_HOST"]
    for key in required:
        if not config.get(key):
            logging.error(f"❌ Missing config value: {key}")
            return None

    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
    return config


# --------------------------------------------------
# 2. PROCESS PDFs (FOLDER → COMPANY)
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

    # Each folder = one company
    for company_dir in base_path.iterdir():
        if not company_dir.is_dir():
            continue

        company_name = company_dir.name
        collection_name = company_name.lower().replace(" ", "_")
        company_docs[collection_name] = []

        logging.info(f"📂 Processing company: {company_name}")

        # Each PDF = one document
        for pdf_file in company_dir.glob("*.pdf"):
            logging.info(f"📄 Processing PDF: {pdf_file.name}")

            # Step 1: Extract text
            extract_text_from_pdf(str(pdf_file))

            extracted_txt = pdf_file.with_name(f"{pdf_file.stem}_extracted.txt")

            # Step 2: Clean text
            cleaned_text = clean_extracted_text(str(extracted_txt))

            if not cleaned_text:
                logging.warning(f"⚠️ No usable text in {pdf_file.name}")
                continue

            # Step 3: Chunk text
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
# 3. INGEST INTO CHROMA
# --------------------------------------------------
def ingest_into_chroma(company_docs: Dict[str, List[Document]], config: Dict[str, Any]) -> None:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=config["EMBEDDING_MODEL"])

        client = chromadb.HttpClient(
            host=config["CHROMA_CLOUD_HOST"],
            port=443,
            ssl=True,
            headers={"X-Chroma-Token": config["CHROMA_CLOUD_API_KEY"]},
        )

    except Exception as e:
        logging.error(f"❌ Failed to initialize Chroma client: {e}")
        return

    for collection_name, docs in company_docs.items():
        if not docs:
            logging.warning(f"⚠️ No documents for company: {collection_name}")
            continue

        logging.info(f"🚀 Ingesting {len(docs)} chunks into '{collection_name}'")

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

        logging.info(f"✅ Ingestion complete for {collection_name}")


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------
def main():
    logging.info("==== LOCAL PDF INGESTION (NO JSON) STARTED ====")

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
