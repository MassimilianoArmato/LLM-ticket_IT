from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Carica documenti
docs = []
for filename in os.listdir("knowledge_base"):
    if filename.endswith(".txt") or filename.endswith(".md"):
        loader = TextLoader(f"knowledge_base/{filename}")
        docs.extend(loader.load())

# Split
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# Embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Indicizza
db = FAISS.from_documents(documents, embedding_model)
db.save_local("faiss_index")