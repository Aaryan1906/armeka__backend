from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, json, random , asyncio , re
from concurrent.futures import ThreadPoolExecutor 

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore_index", embeddings, allow_dangerous_deserialization=True)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

# Request models
class QueryRequest(BaseModel):
    query: str
    is_voice: bool = False  # <-- Flag to indicate if voice input

class ValidateRequest(BaseModel):
    query: str
    expected: str

executor = ThreadPoolExecutor()


@app.post("/ask-text")
async def ask_text(data: QueryRequest):
    query = data.query.strip().lower()
    print(f"🔍 Query received: {query} (voice: {data.is_voice})")

    # Exit handling
    exit_phrases = {"exit", "quit", "close", "stop", "goodbye", "bye"}
    if any(phrase in query for phrase in exit_phrases):
        closing = "👋 Okay, take care! I'm here whenever you need me."
        return {"answer": closing, "voice": closing if data.is_voice else None}

    # Safe greeting match using word boundaries
    greetings = {
        "hi": "Hi there! 👋",
        "hello": "Hello! 😊",
        "hey": "Hey! What's up?",
        "how are you": "I'm just a bot, but I'm running great! 😄",
        "what's up": "Just hanging around, ready to help!",
        "good morning": "Good morning! Ready to learn something new?",
        "good evening": "Good evening! How can I assist you?",
    }
    for key in greetings:
        # Match full words only to avoid bugs like "hi" in "clarity"
        if re.search(rf'\b{re.escape(key)}\b', query):
            response = greetings[key]
            return {"answer": response, "voice": response if data.is_voice else None}

    # Semantic QA (default fallback)
    answer = qa.run(query)
    return {"answer": answer, "voice": answer if data.is_voice else None}


@app.get("/chapters")
def get_chapters():
    try:
        with open("data/handbook.txt", "r", encoding="utf-8") as f:
            chapters = json.load(f)
        return chapters
    except Exception as e:
        return {"error": f"❌ Failed to load chapters: {e}"}

@app.post("/command")
async def process_command(data: QueryRequest):
    query = data.query.lower().strip()
    print(f"🎙 Voice command: {query}")

    try:
        with open("data/handbook.txt", "r", encoding="utf-8") as f:
            chapters = json.load(f)
    except:
        return {"error": "❌ Failed to load chapters"}

    # Teach Chapter
    if "teach" in query and "chapter" in query:
        try:
            num = int("".join(filter(str.isdigit, query)))
            chapter = next((c for c in chapters if c["id"] == f"chapter{num}"), None)
            if chapter:
                content_lines = []
                for section, items in chapter["content"].items():
                    content_lines.append(f"{section}:\n" + "\n".join(items))
                combined = "\n\n".join(content_lines)
                return {
                    "mode": "teach",
                    "chapterId": chapter["id"],
                    "title": chapter["title"],
                    "content": combined,
                    "voice": combined if data.is_voice else None
                }
        except:
            return {"error": "Invalid chapter number in command"}

    # Quiz
    elif any(p in query for p in ["ask", "question", "quiz"]) and "chapter" in query:
        try:
            num = int("".join(filter(str.isdigit, query)))
            chapter = next((c for c in chapters if c["id"] == f"chapter{num}"), None)
            if chapter and "questions" in chapter:
                selected = random.choice(chapter["questions"])
                return {
                    "mode": "quiz",
                    "chapterId": chapter["id"],
                    "title": chapter["title"],
                    "question": selected,
                    "voice": selected if data.is_voice else None
                }
        except:
            return {"error": "Invalid chapter number in command"}

    return {"error": "Unknown command. Try: 'teach chapter 1' or 'ask questions from chapter 2'"}

@app.post("/validate")
async def validate_answer(data: ValidateRequest):
    user = data.query.lower()
    expected = data.expected.lower()

    match = sum(1 for word in expected.split() if word in user)
    score = match / len(expected.split())

    if score >= 0.5:
        return {"result": "✅ Correct!"}
    else:
        return {"result": "❌ Not quite right. Try again!"}