# utils.py
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def split_documents(documents, chunk_size=1500, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
