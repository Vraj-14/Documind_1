import tomli
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ---------------- CONFIG ----------------
def load_config():
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomli.load(f)

    return {
        "GROQ_API_KEY": secrets["GROQ_API_KEY"],
        "CHROMA_API_KEY": secrets["CHROMA_API_KEY"],
        "CHROMA_TENANT": secrets["CHROMA_TENANT"],
        "CHROMA_DATABASE": secrets["CHROMA_DATABASE"],
    }

config = load_config()

# ---------------- INIT EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- INIT CHROMA ----------------
client_chroma = chromadb.CloudClient(
    api_key=config["CHROMA_API_KEY"],
    tenant=config["CHROMA_TENANT"],
    database=config["CHROMA_DATABASE"],
)

# ---------------- INIT GROQ ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=config["GROQ_API_KEY"],
)

# ---------------- FASTAPI ----------------
app = FastAPI(title="Documind RAG API")

class QueryRequest(BaseModel):
    question: str
    company_name: str

# ---------------- RETRIEVE ----------------
def retrieve_context(company_name: str, question: str):

    collection_name = company_name.lower().replace(" ", "_")

    vectorstore = Chroma(
        client=client_chroma,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    return "\n\n".join([doc.page_content for doc in docs])

# ---------------- ENDPOINT ----------------
@app.post("/chat")
async def chat(request: QueryRequest):

    try:
        context = retrieve_context(request.company_name, request.question)

        prompt = f"""
You are a financial document assistant.

Answer strictly using the context below.
If answer is not found, say:
"I could not find the information in the provided documents."

Context:
{context}

Question:
{request.question}
"""

        response = llm.invoke(prompt)

        return {"answer": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

