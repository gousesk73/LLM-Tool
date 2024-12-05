from fastapi import FastAPI, UploadFile, File, HTTPException, Request,Body
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy import create_engine, Column, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
import uuid
import logging
import uvicorn
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
import pdfplumber
import re
from pydantic import BaseModel
import faiss
import pickle

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database Models
class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, index=True)
    model_name = Column(String, index=True)
    api_key = Column(String)

# class ExtractedText(Base):
#     __tablename__ = "extracted_texts"
#     id = Column(String, primary_key=True, index=True)
#     store_id = Column(String, index=True)
#     text_chunk = Column(Text)
#     embedding = Column(Text)
#     streams = Column(Text)
#     units = Column(String)

class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id = Column(String, primary_key=True, index=True)
    title = Column(String, index=True)
    store_id = Column(String, index=True)
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"))
    sender = Column(String)
    text = Column(Text)
    feedback = Column(String, nullable=True)
    parent_message_id = Column(String, nullable=True)  # New column to link responses to questions
    conversation = relationship("Conversation", back_populates="messages")


class FineTuningData(Base):
    __tablename__ = "fine_tuning_data"
    id = Column(String, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"))

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Feedback model for requests
class FeedbackRequest(BaseModel):
    messageId: str  # Ensure this matches the message ID in your frontend
    feedback: str    # Feedback should be "positive" or "negative"

    class Config:
        schema_extra = {
            "example": {
                "messageId": "some-message-id",
                "feedback": "positive"
            }
        }


VECTORSTORE_PATH = "vectorstore.faiss"
METADATA_PATH = "vectorstore_metadata.pkl"

# Initialize FAISS index and metadata store
if os.path.exists(VECTORSTORE_PATH) and os.path.exists(METADATA_PATH):
    # Load existing FAISS index and metadata
    index = faiss.read_index(VECTORSTORE_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    # Create a new FAISS index and metadata
    index = faiss.IndexFlatL2(768)  # Assuming 768-dimensional embeddings
    metadata = {}  # Store metadata for each vector (text_chunk, streams, units)


# Function to extract text from PDF and tables for further processing
async def get_pdf_content(pdf_docs: List[UploadFile]) -> str:
    text_content = ""
    for pdf in pdf_docs:
        try:
            await pdf.seek(0)
            with pdfplumber.open(pdf.file) as pdf_reader:
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text_content += page_text

                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "\n".join(["\t".join([cell or "" for cell in row]) for row in table if row])
                        text_content += "\n" + table_text

            if not text_content.strip():
                logging.warning(f"No text extracted from PDF: {pdf.filename}")

        except Exception as e:
            logging.error(f"Error processing PDF file: {pdf.filename}, Error: {str(e)}")
            continue

    if not text_content.strip():
        logging.error("No text extracted from the provided PDFs.")
        raise HTTPException(status_code=400, detail="No text extracted from the provided PDFs.")

    return text_content

# Function to split extracted text for embedding creation with metadata
def get_text_chunks_with_metadata(text: str) -> List[Dict[str, str]]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    chunk_data = []
    for chunk in chunks:
        stream_numbers = re.findall(r'\b\d{3}\b', chunk)  # Example to find 3-digit stream numbers
        temp_values = re.findall(r'(\d+)\s*°F', chunk)  # Example to find temperature values with °F
        chunk_data.append({
            "text_chunk": chunk,
            "streams": stream_numbers,
            "units": "°F" if "°F" in chunk else None
        })
    return chunk_data

# Embedding generation and storage with metadata
def create_and_save_vectorstore(session, text_chunks: List[Dict[str, str]]) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")
    store_id = str(uuid.uuid4())
    metadata[store_id] = []  # Initialize metadata list for the store_id

    for data in text_chunks:
        try:
            embedding = embeddings.embed_query(data["text_chunk"])
            embedding = np.array(embedding, dtype="float32")
            
            # Add to FAISS index
            index.add(embedding.reshape(1, -1))  # Add single embedding
            
            # Add metadata
            metadata[store_id].append({
                "text_chunk": data["text_chunk"],
                "streams": data["streams"],
                "units": data["units"]
            })
        except Exception as e:
            logging.error(f"Error generating embedding for chunk: {data['text_chunk'][:30]}... - {str(e)}")

    # Save FAISS index and metadata to disk
    faiss.write_index(index, VECTORSTORE_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logging.info(f"Embeddings successfully stored for store_id: {store_id}")
    return store_id

# Query FAISS for relevant chunks
def query_vectorstore(session, store_id: str, user_question: str, user_api_key: str, model_name: str) -> List[str]:
    if "gemini" in model_name.lower():
        embeddings = GoogleGenerativeAIEmbeddings(api_key=user_api_key, model="models/embedding-001")
    else:
        embeddings = OpenAIEmbeddings(api_key=user_api_key)

    question_embedding = embeddings.embed_query(user_question)
    question_embedding = np.array(question_embedding, dtype="float32").reshape(1, -1)

    # Perform similarity search
    distances, indices = index.search(question_embedding, k=3)  # Retrieve top 3 results

    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # Skip invalid indices
        chunk_metadata = metadata[store_id][idx]
        relevant_chunks.append(chunk_metadata["text_chunk"])

    return relevant_chunks


@app.post("/process-pdfs/")
async def process_pdfs(files: List[UploadFile] = File(...)):
    session = SessionLocal()
    try:
        raw_text = ""

        for pdf in files:
            try:
                text = await get_pdf_content([pdf])
                raw_text += text
            except Exception as e:
                logging.error(f"Error processing PDF file: {pdf.filename}, Error: {str(e)}")
                continue

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No valid PDFs processed.")

        # Split text and create chunks with metadata
        text_chunks = get_text_chunks_with_metadata(raw_text)

        # Create and save vectorstore, returns the store_id
        store_id = create_and_save_vectorstore(session, text_chunks)

        # Log the store_id that was created
        logging.info(f"New store_id created: {store_id}")

        return {"message": "PDFs processed and stored successfully", "store_id": store_id}

    except Exception as e:
        logging.error(f"Error during PDF processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")
    finally:
        session.close()


@app.post("/ask-question/")
async def ask_question(request: Request):
    session = SessionLocal()
    form = await request.form()

    # Extract form fields and log the data
    store_id = form.get("store_id")
    user_question = form.get("user_question")
    user_api_key = form.get("user_api_key")
    model_name = form.get("model_name")
    conversation_id = form.get("conversation_id")

    # Log incoming data for debugging purposes
    logging.info(f"Received form data: store_id={store_id}, user_question={user_question}, user_api_key={user_api_key}, model_name={model_name}, conversation_id={conversation_id}")

    # Validate required fields
    if not all([store_id, user_question, user_api_key, model_name, conversation_id]):
        missing_fields = [field for field in ["store_id", "user_question", "user_api_key", "model_name", "conversation_id"] if not form.get(field)]
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")

    # Continue with the rest of the logic
    try:
        logging.info(f"Received question: {user_question} for conversation: {conversation_id}")

        # Save user question to messages table with sender="user"
        question_id = str(uuid.uuid4())
        user_message = Message(
            id=question_id,
            conversation_id=conversation_id,
            sender="user",
            text=user_question
        )
        session.add(user_message)
        session.commit()
        logging.info(f"User message saved with ID: {question_id}")

        # Process the question and retrieve relevant chunks
        relevant_chunks = query_vectorstore(session, store_id, user_question, user_api_key, model_name)

        if not relevant_chunks:
            logging.warning("No relevant documents found.")
            return {"response": "No relevant documents found for your question.", "message_id": question_id}

        # Prepare the prompt for the AI model
        context = "\n\n".join(relevant_chunks)

        prompt = f"""
        You are an advanced AI designed for technical document analysis and precise question answering. Use the provided document excerpts to respond accurately and thoroughly to the user's query. Your response should be detailed, well-structured, and broken into clear sections with numbered or bulleted lists where applicable. Ensure the response is easy to read, with each section focusing on a specific aspect of the answer.

        When providing explanations, please:

        1. Break down complex concepts into easily digestible points.
        2. Use bullet points or numbered lists to present information clearly.
        3. Avoid long paragraphs and ensure each point or section is separated for clarity.
        4. Provide examples when applicable to enhance understanding.

        ### Document Excerpts:
        {context}

        ### User Question:
        {user_question}
        """

        logging.info("Prompt prepared for AI model")

        # Generate the bot's response using the appropriate model
        if "gemini" in model_name.lower():
            model = ChatGoogleGenerativeAI(model=model_name, api_key=user_api_key)
            response = model.invoke(prompt)
            bot_response = response.content if hasattr(response, 'content') else "No answer found."
        else:
            model = ChatOpenAI(model_name=model_name, api_key=user_api_key)
            response = model.invoke(prompt)
            bot_response = response if isinstance(response, str) else "No answer found."

        logging.info(f"Bot response generated: {bot_response}")

        # Save bot response to messages table
        bot_message_id = str(uuid.uuid4())
        bot_message = Message(
            id=bot_message_id,
            conversation_id=conversation_id,
            sender="bot",
            text=bot_response,
            parent_message_id=question_id
        )
        session.add(bot_message)
        session.commit()
        logging.info(f"Bot message saved with ID: {bot_message_id}")

        # Return bot response and message ID
        return {"response": bot_response, "message_id": bot_message_id}
    except Exception as e:
        logging.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error running query")
    finally:
        session.close()



# Additional Routes

@app.post("/save-api-key/")
async def save_api_key(request: Request):
    session = SessionLocal()
    form = await request.form()
    model_name = form.get("model_name")
    api_key = form.get("api_key")

    if not all([model_name, api_key]):
        raise HTTPException(status_code=400, detail="Missing model name or API key.")

    try:
        api_key_entry = APIKey(id=str(uuid.uuid4()), model_name=model_name, api_key=api_key)
        session.add(api_key_entry)
        session.commit()
        return {"message": "API key and model saved successfully"}
    except Exception as e:
        logging.error(f"Error saving API key: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving API key")
    finally:
        session.close()

@app.get("/get-api-key-and-model/")
async def get_api_key_and_model():
    session = SessionLocal()
    try:
        result = session.query(APIKey).first()
        if not result:
            raise HTTPException(status_code=404, detail="No API key or model found")
        return {"model_name": result.model_name, "api_key": result.api_key}
    except Exception as e:
        logging.error(f"Error fetching API key and model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching API key and model")
    finally:
        session.close()

@app.post("/save-conversation/")
async def save_conversation(request: Request):
    session = SessionLocal()
    form = await request.form()
    title = form.get("title")
    store_id = form.get("store_id")

    if not all([title, store_id]):
        raise HTTPException(status_code=400, detail="Missing title or store ID.")

    try:
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(conversation_id=conversation_id, title=title, store_id=store_id)
        session.add(conversation)
        session.commit()
        return {"message": "Conversation created", "conversation_id": conversation_id}
    except Exception as e:
        logging.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating conversation")
    finally:
        session.close()

@app.post("/save-message/")
async def save_message(conversation_id: str, sender: str, text: str):
    session = SessionLocal()
    try:
        if not all([conversation_id, sender, text]):
            raise ValueError("Missing required fields for saving a message.")
        message = Message(id=str(uuid.uuid4()), conversation_id=conversation_id, sender=sender, text=text)
        session.add(message)
        session.commit()
        logging.info(f"Message from {sender} saved for conversation_id: {conversation_id}")
    except Exception as e:
        logging.error(f"Error saving message: {str(e)}")
    finally:
        session.close()

@app.get("/get-all-conversations/")
async def get_all_conversations():
    session = SessionLocal()
    try:
        conversations = session.query(Conversation).with_entities(Conversation.conversation_id, Conversation.title).all()
        return [{"conversation_id": conv.conversation_id, "title": conv.title} for conv in conversations]
    except Exception as e:
        logging.error(f"Error fetching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching conversations")
    finally:
        session.close()

@app.delete("/delete-conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    session = SessionLocal()
    try:
        session.query(Message).filter(Message.conversation_id == conversation_id).delete()
        session.query(Conversation).filter(Conversation.conversation_id == conversation_id).delete()
        session.commit()
        logging.info(f"Deleted conversation with ID: {conversation_id}")
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logging.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")
    finally:
        session.close()

@app.post("/get-conversation-messages/")
async def get_conversation_messages(request: Request):
    session = SessionLocal()
    form = await request.form()
    conversation_id = form.get("conversation_id")

    if not conversation_id:
        raise HTTPException(status_code=400, detail="Missing conversation ID.")

    try:
        messages = session.query(Message).filter(Message.conversation_id == conversation_id).all()
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")

        store_id = session.query(Conversation.store_id).filter(Conversation.conversation_id == conversation_id).scalar()
        return {
            "store_id": store_id,
            "messages": [{"sender": msg.sender, "text": msg.text, "feedback": msg.feedback} for msg in messages]
        }
    except Exception as e:
        logging.error(f"Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")
    finally:
        session.close()

@app.post("/feedback/")
async def feedback(feedback_data: FeedbackRequest = Body(...)):
    logging.info(f"Received feedback data: {feedback_data.dict()}")
    session = SessionLocal()

    try:
        # Validate the feedback value
        if feedback_data.feedback not in ["positive", "negative"]:
            raise HTTPException(status_code=400, detail="Invalid feedback value. Must be 'positive' or 'negative'.")

        # Fetch the message to update
        message_to_update = session.query(Message).filter(Message.id == feedback_data.messageId).first()

        if not message_to_update:
            logging.warning(f"Message ID not found: {feedback_data.messageId}")
            raise HTTPException(status_code=404, detail="Message not found")

        # Update the feedback field
        message_to_update.feedback = feedback_data.feedback
        session.commit()

        logging.info(f"Feedback updated for message ID: {feedback_data.messageId}")

        # If feedback is positive and the message is from the bot, save question-answer pair
        if feedback_data.feedback == "positive" and message_to_update.sender == "bot":
            logging.info("Processing positive feedback for fine-tuning...")

            # Retrieve the parent user message
            parent_message = session.query(Message).filter(
                Message.id == message_to_update.parent_message_id,
                Message.sender == "user"
            ).first()

            if parent_message:
                # Save the question-answer pair in FineTuningData
                fine_tuning_data = FineTuningData(
                    id=str(uuid.uuid4()),
                    question=parent_message.text,
                    answer=message_to_update.text,
                    conversation_id=message_to_update.conversation_id
                )
                session.add(fine_tuning_data)
                session.commit()

                logging.info("Question-answer pair saved successfully for fine-tuning.")
            else:
                logging.warning("Parent user message not found for the bot message.")
        else:
            logging.info("No fine-tuning required: Feedback is not positive or message is not from the bot.")

        return {"message": "Feedback received and processed successfully"}

    except HTTPException as http_err:
        logging.error(f"HTTP error occurred: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        session.close()

# Initialize the database and run the app
if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="127.0.0.1", port=8000)
