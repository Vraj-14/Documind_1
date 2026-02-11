import tomli
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#import tomli
import tomllib
import chromadb

# Load secrets
#with open(".streamlit/secrets.toml", "rb") as f:
with open("E:\8th Sem\MAJOR\Documind\.streamlit\secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

client = chromadb.CloudClient(
    api_key=secrets["CHROMA_API_KEY"],
    tenant=secrets["CHROMA_TENANT"],
    database=secrets["CHROMA_DATABASE"]
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    client=client,
    collection_name="abc",   # use your collection name exactly
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("What is mentioned in this document?")

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
