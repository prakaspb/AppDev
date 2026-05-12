from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ✅ Always resolve paths relative to THIS file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "policy.pdf")
DB_PATH = os.path.join(BASE_DIR, "vectorstore")

print("Current working directory:", os.getcwd())
print("PDF absolute path:", DATA_PATH)
print("PDF exists:", os.path.exists(DATA_PATH))


#DATA_PATH = "data/policy.pdf"
#DB_PATH = "vectorstore"


def ingest_policy():
    print("Loading policy PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    print("Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Saving vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

    print("Policy ingestion completed successfully!")


if __name__ == "__main__":
    ingest_policy()