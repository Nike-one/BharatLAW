import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load documents with logging
logging.info("Loading documents...")
loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents into texts with logging
logging.info("Splitting documents into texts...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize the text splitter and embeddings model with logging
logging.info("Initializing the text splitter and embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})

# Creates vector embeddings and saves it in the FAISS DB with logging
logging.info("Creating vector embeddings and saving to the FAISS DB...")
faiss_db = FAISS.from_documents(texts, embeddings)

# Saves and exports the vector embeddings database with logging
logging.info("Exporting the vector embeddings database...")
faiss_db.save_local("ipc_vector_db")

# Log a message to indicate the completion of the process
logging.info("Process completed successfully.")