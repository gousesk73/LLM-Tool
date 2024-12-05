import streamlit as st
import faiss
import pickle
import numpy as np
import os
import uuid
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Predefined Gemini API Key (replace with your actual API key)
GEMINI_API_KEY = "AIzaSyBj72cio39FXCvAQk5V71LFiD25g3dR0mA"

# Constants for FAISS and metadata paths
VECTORSTORE_PATH = "vectorstore.faiss"
METADATA_PATH = "vectorstore_metadata.pkl"

# Initialize FAISS index and metadata store
if os.path.exists(VECTORSTORE_PATH) and os.path.exists(METADATA_PATH):
    index = faiss.read_index(VECTORSTORE_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(768)  # Assuming 768-dimensional embeddings
    metadata = {}  # Store metadata for each vector (text_chunk, document_name)

# Helper function: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Helper function: Split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Helper function: Process uploaded documents
def process_documents(uploaded_files):
    global index, metadata
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GEMINI_API_KEY, model="models/embedding-001"
    )

    for uploaded_file in uploaded_files:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_into_chunks(pdf_text)

        # Generate embeddings and store them in FAISS
        for chunk in chunks:
            embedding = embeddings.embed_query(chunk)
            embedding = np.array(embedding, dtype="float32").reshape(1, -1)
            index.add(embedding)

            # Store metadata
            doc_id = str(uuid.uuid4())
            metadata[doc_id] = {"text_chunk": chunk, "document_name": uploaded_file.name}

    # Save updated vector store and metadata
    faiss.write_index(index, VECTORSTORE_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

# Helper function: Query FAISS for relevant chunks
def query_vectorstore(question):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GEMINI_API_KEY, model="models/embedding-001"
    )
    question_embedding = embeddings.embed_query(question)
    question_embedding = np.array(question_embedding, dtype="float32").reshape(1, -1)

    # Perform similarity search
    distances, indices = index.search(question_embedding, k=3)  # Top 3 results

    relevant_chunks = []
    for idx in indices[0]:
        if idx == -1:
            continue
        chunk_metadata = metadata[list(metadata.keys())[idx]]
        relevant_chunks.append(chunk_metadata["text_chunk"])

    return relevant_chunks

# Streamlit UI
st.title("AI-Powered Document Analysis and Q&A")
st.sidebar.header("Upload Documents")

# File upload and processing section
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        process_documents(uploaded_files)
        st.sidebar.success("Processing done! Documents added to vector store.")
    else:
        st.sidebar.error("Please upload at least one PDF file.")

# Input section for querying
user_question = st.text_area("Enter your question here:")

if st.button("Submit Question"):
    if not user_question:
        st.error("Please enter a question.")
    else:
        try:
            # Retrieve relevant chunks
            relevant_chunks = query_vectorstore(user_question)

            if not relevant_chunks:
                st.warning("No relevant documents found.")
            else:
                # Prepare context and prompt for the AI model
                context = "\n\n".join(relevant_chunks)
                prompt = f"""
                You are an advanced AI for analyzing technical documents and answering questions based on the provided excerpts. 
                ### Document Excerpts:
                {context}
                ### User Question:
                {user_question}
                """

                # Generate response
                model = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY
                )
                response = model.invoke(prompt)
                bot_response = response.content if hasattr(response, "content") else "No answer found."

                # Display the response
                st.success("Response from the AI:")
                st.write(bot_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
