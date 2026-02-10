import os
import logging
import tomli
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any

# --- 0. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. CONFIGURATION ---
def load_configuration() -> Dict[str, Any]:
    """Loads configuration from secrets.toml and validates required keys."""
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomli.load(f)
    except FileNotFoundError:
        logging.error("Error: .streamlit/secrets.toml not found. Please create it.")
        raise

    config = {
        "GOOGLE_API_KEY": secrets.get("GEMINI_API_KEY"),
        "CHROMA_API_KEY": secrets.get("CHROMA_CLOUD_API_KEY"),
        "CHROMA_CLOUD_HOST": secrets.get("CHROMA_CLOUD_HOST"),
        "EMBEDDING_MODEL": "models/text-embedding-004",
        "LLM_MODEL": "gemini-1.5-flash", # Using a more recent model name
    }

    # More explicit check for truly required keys
    required_keys = ["GOOGLE_API_KEY", "CHROMA_API_KEY", "CHROMA_CLOUD_HOST"]
    missing_keys = [key for key in required_keys if not config.get(key)]

    if missing_keys:
        # Log the specific keys from secrets.toml that are missing
        secrets_map = {"GOOGLE_API_KEY": "GEMINI_API_KEY", "CHROMA_API_KEY": "CHROMA_CLOUD_API_KEY", "CHROMA_CLOUD_HOST": "CHROMA_CLOUD_HOST"}
        missing_secrets = [secrets_map[key] for key in missing_keys]
        for key in missing_keys:
            logging.error(f"Error: '{key}' not found or is empty in secrets.toml.")
        raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")

    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
    logging.info("Configuration loaded successfully.")
    return config

app = FastAPI(title="InstaDocs RAG API")

# --- 2. INITIALIZE AI COMPONENTS ---
try:
    config = load_configuration()
    embeddings = GoogleGenerativeAIEmbeddings(model=config["EMBEDDING_MODEL"])
    client = chromadb.HttpClient(host=config["CHROMA_CLOUD_HOST"], port=443, ssl=True, headers={"X-Chroma-Token": config["CHROMA_API_KEY"]})
    llm = ChatGoogleGenerativeAI(model=config["LLM_MODEL"], temperature=0)
    logging.info("AI components initialized successfully.")
except (ValueError, Exception) as e:
    logging.critical(f"Failed to initialize application: {e}")
    # In a real app, you might exit or prevent the app from starting

# Define Prompt
template = """**Your Role and Goal:**
You are an expert financial analyst and a highly capable document retrieval assistant. Your primary goal is to provide precise, factual, and insightful answers to financial questions based *exclusively* on the provided document context. You must act as a reliable and professional assistant.

**Critical Instructions:**

1.  **Strict Context Adherence:** Your answer MUST be derived *solely* from the information within the `Context` section below. Do not use any external knowledge, make assumptions, or infer information not explicitly stated in the documents.

2.  **Direct and Factual Answers:**
    *   Answer the user's `Question` directly and concisely.
    *   If the context contains numerical data, tables, or specific figures relevant to the question, present them clearly in your answer.
    *   Do not editorialize, offer opinions, or add information that is not present in the context.

3.  **Handling Missing Information:**
    *   If the answer to the `Question` cannot be found within the provided `Context`, you MUST state: "I could not find the information in the provided documents."
    *   Do not apologize or use phrases like "I'm sorry" or "Unfortunately". Do not suggest where the user might find the information.

4.  **Mandatory Citation:**
    *   At the very end of your response, you MUST include the source link.
    *   The link is provided in the context as `Direct Link: [URL]`.
    *   Format the citation on a new line as: `Source: [URL]`

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 3. API MODELS & HELPERS ---
class QueryRequest(BaseModel):
    question: str
    company_name: str

def format_docs(docs):
    formatted = []
    for d in docs:
        content = d.page_content
        link = d.metadata.get('link', 'No Link')
        formatted.append(f"{content}\nDirect Link: {link}")
    return "\n\n".join(formatted)

def get_retriever_for_company(company_name: str):
    """Gets a retriever for a specific company's Chroma collection."""
    try:
        vectorstore = Chroma(client=client, collection_name=company_name, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        # This might happen if the collection doesn't exist.
        logging.error(f"Failed to get retriever for '{company_name}': {e}")
        raise HTTPException(status_code=404, detail=f"No document collection found for company '{company_name}'. Please ingest data first.")

# --- 4. API ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        retriever = get_retriever_for_company(request.company_name.lower().replace(" ", "_"))
        # Build RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(request.question)
        return {"answer": response}
        
    except HTTPException as e:
        raise e # Re-raise HTTPException to keep status code and detail
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)