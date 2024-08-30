import streamlit as st
from time import sleep
from streamlit_mic_recorder import mic_recorder
from streamlit_chat import message
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings
import chromadb
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Configure Chroma to use an in-memory SQLite database
chroma_setting = Settings(
    chroma_db_impl="sqlite",
    persist_directory=None,  # This makes it in-memory
    anonymized_telemetry=False
)

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    newest_file_path = max(paths, key=os.path.getctime)
    return os.path.basename(newest_file_path)

def text_to_audio(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    # Save the audio as an MP3 file
    mp3_file = "temp_audio.mp3"
    tts.save(mp3_file)
    return mp3_file

def save_uploaded_file(uploaded_file, directory):
    try:
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return st.success(f"Saved file: {uploaded_file.name} to {directory}")
    except Exception as e:
        return st.error(f"Error saving file: {e}")

# Create a directory to save the uploaded files
upload_dir = "uploaded_files"
os.makedirs(upload_dir, exist_ok=True)

# Setup the LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.1,
    max_tokens=1000,
)

# Setup the embedding Model
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Setup the text splitter
def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
    )

# RetrievalQA
def answer_question(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": question})
    return result['result']

# Initialize the Groq client
groq_client = Groq()

# Specify the path to the audio file
filename = "recorded_audio.wav"

# Helper Function to Transcribe Audio Recording
def transcribe_audio(filename):
    with open(filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="distil-whisper-large-v3-en",
            prompt="Specify context or spelling",
            response_format="json",
            language="en",
            temperature=0.0
        )
    return transcription.text

# Initialize a session state variable to track if the app should stop
if 'stop' not in st.session_state:
    st.session_state.stop = False

# Set page configuration
st.set_page_config(
    page_title="Audio and Book App",
    page_icon="📚",
    layout="wide"
)

# Create two columns
col1, col2 = st.columns([1, 2])

with col2:
    st.title("PDF Upload and Reader")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Setup the Vectorstore and Add Documents
    if uploaded_file is not None:
        save_uploaded_file(uploaded_file, upload_dir)
        file_name = uploaded_file.name
        loader = PyPDFLoader(f"uploaded_files/{file_name}")
        pages = loader.load_and_split(text_splitter())
        client = chromadb.PersistentClient(settings=chroma_setting)
        vectorstore = Chroma(
            embedding_function=embeddings,
            client=client,
            collection_name=file_name.split(".")[0],
            client_settings=chroma_setting,
        )
        # Load documents into vectorstore
        MAX_BATCH_SIZE = 100
        for i in range(0, len(pages), MAX_BATCH_SIZE):
            batch = pages[i:min(len(pages), i + MAX_BATCH_SIZE)]
            vectorstore.add_documents(batch)

    # Initialize session state variable
    if 'start_process' not in st.session_state:
        st.session_state.start_process = False

    # Create a button to start the process
    if st.button("Start Process"):
        st.session_state.start_process = True

    if st.session_state.start_process:
        options = os.listdir("uploaded_files")
        options += ["none"]
        selected_option = st.selectbox("Select an option:", options)
        if selected_option == "none":
            file_name = newest("uploaded_files")
        else:
            file_name = selected_option
        st.write(f"You selected: {selected_option}")
        st.title("Audio Recorder- Ask Question based on the selected option")

        with st.spinner("Audio Recording in progress..."):
            audio = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                just_once=False,
                key='recorder'
            )
            if audio:
                st.audio(audio['bytes'], format='audio/wav')
                with open("recorded_audio.wav", "wb") as f:
                    f.write(audio['bytes'])
                st.success("Audio Recording is completed!")

        with st.spinner("Transcribing Audio in progress ..."):
            text = transcribe_audio(filename)
            st.markdown(text)

        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat messages from history
        for i, chat in enumerate(st.session_state.chat_history):
            message(chat["question"], is_user=True, key=f"question_{i}")
            message(chat["response"], is_user=False, key=f"response_{i}")

        if text:
            with st.spinner("Synthesizing Response ....."):
                client = chromadb.PersistentClient(settings=chroma_setting)
                vectorstore = Chroma(
                    embedding_function=embeddings,
                    client=client,
                    collection_name=file_name.split(".")[0],
                    client_settings=chroma_setting
                )
                response = answer_question(text, vectorstore)
                st.success("Response Generated")

            aud_file = text_to_audio(response)

            # Add the question and response to chat history
            st.session_state.chat_history.append({"question": text, "response": response})
            message(text, is_user=True)
            message(response, is_user=False)

            st.title("Audio Playback")
            st.audio(aud_file, format='audio/wav', start_time=0)
