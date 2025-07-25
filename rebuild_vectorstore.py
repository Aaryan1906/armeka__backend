from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load your company handbook from the data folder
loader = TextLoader("data/handbook.txt")  # ✅ Make sure this file exists
documents = loader.load()

# Use sentence-transformer embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Save it locally for chatbot use
vectorstore.save_local("vectorstore_index")

print("✅ Vectorstore successfully built and saved.")

