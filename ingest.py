import json
import os
import logging
import tomli
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Tuple, Optional

# --- 0. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. CONFIGURATION ---
def load_configuration():
    """Loads configuration from secrets.toml and validates required keys."""
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomli.load(f)
    except FileNotFoundError:
        logging.error("Error: .streamlit/secrets.toml not found. Please create it.")
        return None

    config = {
        "GOOGLE_API_KEY": secrets.get("GEMINI_API_KEY"),
        "CHROMA_CLOUD_API_KEY": secrets.get("CHROMA_CLOUD_API_KEY"),
        "CHROMA_CLOUD_HOST": secrets.get("CHROMA_CLOUD_HOST"),
        "JSON_FILE": secrets.get("general", {}).get("json_file_name"),
        "EMBEDDING_MODEL": "models/text-embedding-004",
        "CHUNK_SIZE": 1000,
        "CHUNK_OVERLAP": 200,
    }

    required_keys = ["GOOGLE_API_KEY", "CHROMA_CLOUD_API_KEY", "CHROMA_CLOUD_HOST", "JSON_FILE"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    if missing_keys:
        # Log the specific keys from secrets.toml that are missing
        secrets_map = {"GOOGLE_API_KEY": "GEMINI_API_KEY", "CHROMA_CLOUD_API_KEY": "CHROMA_CLOUD_API_KEY", "CHROMA_CLOUD_HOST": "CHROMA_CLOUD_HOST", "JSON_FILE": "json_file_name under [general]"}
        for key in missing_keys:
            logging.error(f"Error: '{secrets_map.get(key, key)}' not found or is empty in secrets.toml.")
        return None # Exit if required keys are missing
    
    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
    return config

def fetch_document_content(url: str) -> str:
    """
    Placeholder function to fetch and extract text from a document URL.
    In a real implementation, this would handle various document types (PDF, etc.)
    and include robust error handling for network issues or parsing failures.
    """
    logging.info(f"Fetching content for {url} (placeholder)...")
    return "This is placeholder content for the document. In a real implementation, you would fetch and parse the actual document content from the URL."

def load_company_and_documents(json_file: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Loads company name and document list from the specified JSON file."""
    logging.info(f"Loading documents from {json_file}...")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        company_name = data.get('InstaAPIReport', {}).get('Metadata', {}).get('CompanyName')
        if not company_name:
            logging.error("CompanyName not found in JSON metadata.")
            return None, []

        docs_list = data.get('InstaAPIReport', {}).get('Reportdata', {}).get('InstaDocs', {}).get('Document', [])
        return company_name, docs_list
    except FileNotFoundError:
        logging.error(f"Error: File '{json_file}' not found.")
        return None, []
    except KeyError as e:
        logging.error(f"Error parsing JSON structure: {e}")
        return None, []

def process_and_chunk_documents(company_name: str, docs_list: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> Dict[str, List[Document]]:
    """Processes documents, fetches content, and splits them into chunks."""
    logging.info(f"Processing and chunking {len(docs_list)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    company_docs_for_chroma = {}

    for item in docs_list:
        document_link = item.get('DocumentLink')
        collection_name = company_name.lower().replace(" ", "_")
        if document_link:
            try:
                # In a real scenario, you'd fetch and parse the actual content.
                # page_content = fetch_document_content(document_link)
                page_content = (
                    f"Document Name: {item.get('DocumentName', 'Unknown')}. "
                    f"Category: {item.get('DocumentCategory', 'Unknown')}. "
                    f"Filing Date: {item.get('DocumentFillingDate', 'Unknown')}."
                )
                chunks = text_splitter.split_text(page_content)

                if collection_name not in company_docs_for_chroma:
                    company_docs_for_chroma[collection_name] = []

                for chunk in chunks:
                    metadata = {
                        "doc_name": item.get('DocumentName', 'Unknown'),
                        "category": item.get('DocumentCategory', 'Unknown'),
                        "date": item.get('DocumentFillingDate', 'Unknown'),
                        "link": item.get('DocumentLink', 'N/A'),
                        "company": collection_name
                    }
                    company_docs_for_chroma[collection_name].append(Document(page_content=chunk, metadata=metadata))
            except Exception as e:
                logging.error(f"Failed to process document {item.get('DocumentName')}: {e}")
        else:
            logging.warning(f"Skipping document with no link: {item.get('DocumentName')}")
    
    return company_docs_for_chroma

def ingest_into_chroma(company_docs: Dict[str, List[Document]], config: Dict[str, Any]):
    """Initializes clients and ingests documents into ChromaDB collections."""
    logging.info("Initializing embeddings and ChromaDB client...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=config["EMBEDDING_MODEL"])
        client = chromadb.HttpClient(
            host=config["CHROMA_CLOUD_HOST"],
            port=443,
            ssl=True,
            headers={"X-Chroma-Token": config["CHROMA_CLOUD_API_KEY"]}
        )
    except Exception as e:
        logging.error(f"Failed to initialize clients: {e}")
        return

    for company_name, docs in company_docs.items():
        if not docs:
            logging.warning(f"No documents to ingest for company: {company_name}")
            continue
        
        logging.info(f"Ingesting {len(docs)} document chunks for company: {company_name}")
        try:
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                client=client,
                collection_name=company_name,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logging.info(f"Success! Ingested documents for {company_name} into Chroma Cloud.")
        except Exception as e:
            logging.error(f"Failed to ingest documents for {company_name}: {e}")

def search_documents_in_json(docs_list: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """
    Searches for documents in the loaded JSON data that match a keyword.
    The search is case-insensitive and checks the document name and category.
    """
    if not keyword:
        logging.warning("Search keyword is empty, returning all documents.")
        return docs_list

    logging.info(f"Searching for documents containing the keyword: '{keyword}'")
    keyword_lower = keyword.lower()
    
    found_docs = [
        doc for doc in docs_list 
        if keyword_lower in doc.get('DocumentName', '').lower() 
        or keyword_lower in doc.get('DocumentCategory', '').lower()
    ]
    
    return found_docs

def main():
    """Main function to run the data ingestion process."""
    logging.info("--- Script Started ---")
    config = load_configuration()
    if not config:
        logging.error("Configuration loading failed. Exiting.")
        return

    company_name, docs_list = load_company_and_documents(config["JSON_FILE"])
    if not company_name or not docs_list:
        logging.error("No company or documents loaded. Exiting.")
        return

    # --- Search directly in the JSON data instead of ingesting ---
    search_keyword = "Annual Report" # Example: change this to search for different documents
    search_results = search_documents_in_json(docs_list, search_keyword)

    if search_results:
        logging.info(f"Found {len(search_results)} documents matching '{search_keyword}':")
        for doc in search_results:
            print(f"  - Name: {doc.get('DocumentName')}, Category: {doc.get('DocumentCategory')}, Date: {doc.get('DocumentFillingDate')}")
    else:
        logging.warning(f"No documents found matching '{search_keyword}'.")

    # --- The following code for ChromaDB ingestion is now commented out ---
    # logging.info("--- Ingestion Process Started ---")
    # company_docs = process_and_chunk_documents(company_name, docs_list, config["CHUNK_SIZE"], config["CHUNK_OVERLAP"])
    # ingest_into_chroma(company_docs, config)
    # logging.info("--- Ingestion Process Finished ---")
    logging.info("--- Script Finished ---")

if __name__ == "__main__":
    main()