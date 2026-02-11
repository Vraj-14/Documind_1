import os
import logging
import tomli
import chromadb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# 0. LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Documind RAG API")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
def load_configuration() -> Dict[str, Any]:
    try:
        with open(".streamlit/secrets.toml", "rb") as f:
            secrets = tomli.load(f)
    except FileNotFoundError:
        raise RuntimeError("❌ .streamlit/secrets.toml not found")

    config = {
        "GEMINI_API_KEY": secrets.get("GEMINI_API_KEY"),
        "CHROMA_API_KEY": secrets.get("CHROMA_API_KEY"),
        "CHROMA_TENANT": secrets.get("CHROMA_TENANT"),
        "CHROMA_DATABASE": secrets.get("CHROMA_DATABASE"),
        "EMBEDDING_MODEL": "text-embedding-004",
        "LLM_MODEL": "gemini-1.5-flash",
    }

    required = ["GEMINI_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"]

    for key in required:
        if not config.get(key):
            raise ValueError(f"❌ Missing config value: {key}")

    os.environ["GEMINI_API_KEY"] = config["GEMINI_API_KEY"]

    logging.info("✅ Configuration loaded successfully")
    return config


# --------------------------------------------------
# 2. INITIALIZE COMPONENTS
# --------------------------------------------------
try:
    config = load_configuration()

    # Embeddings (MUST match ingestion model)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["EMBEDDING_MODEL"]
    )

    # Chroma Cloud Client
    client = chromadb.CloudClient(
        api_key=config["CHROMA_API_KEY"],
        tenant=config["CHROMA_TENANT"],
        database=config["CHROMA_DATABASE"],
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config["LLM_MODEL"],
        temperature=0,
    )

    logging.info("✅ AI components initialized successfully")

except Exception as e:
    logging.critical(f"❌ Failed to initialize backend: {e}")
    raise


# --------------------------------------------------
# 3. PROMPT TEMPLATE
# --------------------------------------------------
template = """
You are a financial document analysis assistant.

Answer the question strictly using the provided context.
If the answer is not in the context, say:
"I could not find the information in the provided documents."

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


# --------------------------------------------------
# 4. REQUEST MODEL
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    company_name: str


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_retriever_for_company(company_name: str):
    try:
        vectorstore = Chroma(
            client=client,
            collection_name=company_name,
            embedding_function=embeddings,
        )

        return vectorstore.as_retriever(search_kwargs={"k": 5})

    except Exception as e:
        logging.error(f"❌ Collection not found: {company_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No collection found for company '{company_name}'.",
        )


# --------------------------------------------------
# 6. API ENDPOINT
# --------------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        collection_name = request.company_name.lower().replace(" ", "_")
        retriever = get_retriever_for_company(collection_name)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(request.question)

        return {"answer": response}

    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# 7. RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
