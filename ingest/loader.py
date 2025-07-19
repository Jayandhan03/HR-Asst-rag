import os
import logging
from typing import Generator, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFStreamingLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def stream_documents(self) -> Generator[Document, None, None]:
        """Streams individual pages from all PDFs in the folder with rich metadata."""
        if not os.path.exists(self.folder_path):
            logger.error(f"Folder not found: {self.folder_path}")
            return

        for filename in os.listdir(self.folder_path):
            if not filename.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(self.folder_path, filename)
            logger.info(f"📄 Processing: {filename}")

            try:
                loader = PyPDFLoader(full_path)
                docs = loader.load()  # Each doc = 1 page
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {e}")
                continue

            if not docs:
                logger.warning(f"⚠️ Empty or unreadable PDF: {filename}")
                continue

            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "file_path": full_path,
                    "page_number": doc.metadata.get("page", -1),
                    "type": "startup_report",
                })
                yield doc

    def load_all(self) -> List[Document]:
        """Eagerly loads all documents into memory (not recommended for huge PDFs)."""
        return list(self.stream_documents())
