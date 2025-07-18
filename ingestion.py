import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# LangChain document loading and splitting
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-n8n-demo")

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,  # Gemini embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Load and split PDF documents
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)

# Generate unique IDs
uuids = [f"id{i+1}" for i in range(len(documents))]

# Get embeddings for each chunk using Gemini
chunks = [doc.page_content for doc in documents]
result = genai.embed_content(
    model="gemini-embedding-001",
    content=chunks
)
embeddings = result["embedding"]

# Prepare metadata for each chunk
metadatas = [
    {**doc.metadata, "text": doc.page_content}
    for doc in documents
]

# Add to Pinecone with metadata
index.upsert(
    vectors=[
        {
            "id": uuids[i],
            "values": embeddings[i],
            "metadata": metadatas[i]
        }
        for i in range(len(uuids))
    ]
)