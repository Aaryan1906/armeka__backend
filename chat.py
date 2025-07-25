import os
from dotenv import load_dotenv  # NEW
import tempfile
import threading
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 🌱 Load environment variables from .env
load_dotenv()

# 📄 Load documents
loader = TextLoader("data/handbook.txt")
documents = loader.load()

# ✂️ Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# 🧠 Vector DB
embedding = OpenAIEmbeddings()
vectordb = FAISS.from_documents(splits, embedding)
retriever = vectordb.as_retriever()

# 🔍 QA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# 🎤 Listen to microphone
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        print("❌ Could not understand.")
        return ""
    except sr.RequestError:
        print("❌ Speech recognition error.")
        return ""

# 🧠 Interrupt check
def monitor_interrupt(interrupt_event):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            interrupt_event.set()
        except sr.WaitTimeoutError:
            pass

# 🔈 Speak with interrupt
def speak(text, allow_interrupt=True):
    tts = gTTS(text=text, lang='en', tld='co.in')  # Indian accent
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio = AudioSegment.from_mp3(fp.name)
        play_obj = _play_with_simpleaudio(audio)

        if allow_interrupt:
            interrupt_event = threading.Event()
            mic_thread = threading.Thread(target=monitor_interrupt, args=(interrupt_event,))
            mic_thread.start()

            while play_obj.is_playing():
                if interrupt_event.is_set():
                    play_obj.stop()
                    print("🔇 Interrupted by user.")
                    break
            mic_thread.join()
        else:
            play_obj.wait_done()

# 👋 Handle greetings
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "thanks", "thank you"]
    return text.lower().strip() in greetings

# 🧠 Main app loop
def main():
    mode = input("Use voice or text? (v/t): ").strip().lower()
    print("Type or say 'exit' to quit.")

    greeting = "Welcome to Armeka Assistant. How can I help you today?"

    if mode == "v":
        print(f"🤖 ArmekaBot: {greeting}")
        speak(greeting, allow_interrupt=False)
    else:
        print(f"🤖 ArmekaBot: {greeting}")

    while True:
        if mode == "v":
            query = listen()
        else:
            query = input("You: ")

        if query.lower().strip() in ["exit", "bye", "close"]:
            print("👋 Goodbye!")
            if mode == "v":
                speak("Goodbye!", allow_interrupt=False)
            break

        if is_greeting(query):
            response = "Hello! How can I assist you?"
        else:
            docs = retriever.get_relevant_documents(query)
            if not docs:
                response = "Sorry, I can only help with questions related to the ZevoTech employee handbook."
            else:
                response = qa.run(query)

        print(f"🤖 ArmekaBot: {response}")
        if mode == "v":
            speak(response)

if __name__ == "__main__":
    main()
