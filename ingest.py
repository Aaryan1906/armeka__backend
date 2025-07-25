import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def extract_json_handbook(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    documents = []

    for chapter in chapters:
        parts = [f"Title: {chapter['title']}", f"Description: {chapter['description']}"]

        # Flatten content sections
        if "content" in chapter:
            for section_title, lines in chapter["content"].items():
                parts.append(f"{section_title}:")
                for line in lines:
                    parts.append(f"- {line}")

        # Add questions and answers
        if "questions" in chapter and "answers" in chapter:
            for q, a in zip(chapter["questions"], chapter["answers"]):
                parts.append(f"Q: {q}")
                parts.append(f"A: {a}")

        content = "\n".join(parts)
        documents.append(Document(page_content=content))

    return documents

def split_and_embed(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore_index")
    print("✅ Vector store saved.")

if __name__ == "__main__":
    docs = extract_json_handbook("data/handbook.txt")
    split_and_embed(docs)
    print("✅ Ingestion complete.")
