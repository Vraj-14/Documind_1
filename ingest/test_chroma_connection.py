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
    database=secrets["CHROMA_DATABASE"],
)

print("Connected successfully!")
print("Existing collections:", client.list_collections())
