from pathlib import Path
from typing import List
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbase.config import Config


class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="interquartile"
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
            add_start_index=True,
        )

    def extract_text_from_image(self, image_path: Path) -> str:
        """Extracts text from an image using OCR (Tesseract)."""
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        documents = []
        for doc_path in doc_paths:
            # Check if the file is a PDF or image
            if doc_path.suffix.lower() == '.pdf':
                # Use PyPDFium2Loader to load and extract text from PDF
                loaded_documents = PyPDFium2Loader(doc_path).load()
                document_text = "\n".join([doc.page_content for doc in loaded_documents])

            elif doc_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # If it's an image, use OCR to extract text
                document_text = self.extract_text_from_image(doc_path)

            else:
                continue  # Skip unsupported file types

            # Split the document text into manageable chunks
            documents.extend(
                self.recursive_splitter.split_documents(
                    self.semantic_splitter.create_documents([document_text])
                )
            )

        # Index the documents into Qdrant
        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=Config.Path.DATABASE_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )
