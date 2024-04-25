import ray
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2  # Assuming using L2 distance for simplicity

# Initialize Ray
ray.init()

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load documents with logging
logging.info("Loading documents...")
loader = DirectoryLoader('data', glob="./*.txt")
documents = loader.load()

# Extract text from documents and split into manageable texts with logging
logging.info("Extracting and splitting texts from documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = []
for document in documents:
    if hasattr(document, 'get_text'):
        text_content = document.get_text()  # Adjust according to actual method
    else:
        text_content = ""  # Default to empty string if no text method is available

    texts.extend(text_splitter.split_text(text_content))

# Define embedding function
def embedding_function(text):
    embeddings_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    return embeddings_model.embed_query(text)

# Create FAISS index for embeddings
index = IndexFlatL2(768)  # Dimension of embeddings, adjust as needed

# Assuming docstore as a simple dictionary to store document texts
docstore = {i: text for i, text in enumerate(texts)}
index_to_docstore_id = {i: i for i in range(len(texts))}

# Initialize FAISS
faiss_db = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Process and store embeddings
logging.info("Storing embeddings in FAISS...")
for i, text in enumerate(texts):
    embedding = embedding_function(text)
    faiss_db.add_documents([embedding])

# Exporting the vector embeddings database with logging
logging.info("Exporting the vector embeddings database...")
faiss_db.save_local("ipc_embed_db")

# Log a message to indicate the completion of the process
logging.info("Process completed successfully.")

# Shutdown Ray after the process
ray.shutdown()
