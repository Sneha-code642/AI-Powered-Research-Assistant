import streamlit as st
import os
import openai
import tempfile
import requests
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

# LangChain & FAISS
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ResearchAssistant:
    def __init__(self):
        self.api_key = self._load_api_key()
        if self.api_key:
            openai.api_key = self.api_key
            os.environ["OPENAI_API_KEY"] = self.api_key
        self.vectorstore = None

    def _load_api_key(self):
        """Load OpenAI API key from .env or .config"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                with open(".config") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                st.error("⚠️ OpenAI API key not found. Please set it in .env or .config.")
                return None
        return api_key

    def is_pdf_url(self, url):
        """Check if URL points to a PDF"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = response.headers.get("content-type", "").lower()
            return "application/pdf" in content_type or url.lower().endswith(".pdf")
        except Exception as e:
            logger.warning(f"PDF check failed for {url}: {str(e)}")
            return False

    def download_pdf(self, url):
        """Download PDF from URL to a temp file"""
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return None

    def process_url(self, url):
        """Load documents from a single URL"""
        try:
            logger.info(f"Processing URL: {url}")
            if self.is_pdf_url(url):
                pdf_path = self.download_pdf(url)
                if pdf_path:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    os.unlink(pdf_path)  # Cleanup
                    return docs
            else:
                loader = UnstructuredURLLoader(urls=[url])
                return loader.load()
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            st.warning(f"⚠️ Could not process {url}")
            return []

    def process_urls(self, urls):
        """Load & split documents from multiple URLs"""
        documents = []
        for url in urls:
            if url.strip():
                documents.extend(self.process_url(url.strip()))

        if not documents:
            st.error("❌ No valid documents found.")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        return splitter.split_documents(documents)

    def create_embeddings(self, docs):
        """Generate FAISS embeddings"""
        try:
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(docs, embeddings)
            self.vectorstore.save_local("faiss_index")  # Auto-generates folder
            return True
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            st.error(f"❌ Failed to create embeddings: {str(e)}")
            return False

    def get_answer(self, query):
        """Answer queries using RAG pipeline"""
        try:
            if not self.vectorstore:
                self.vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings())

            llm = OpenAI(temperature=0)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=self.vectorstore.as_retriever()
            )
            return chain({"question": query})
        except Exception as e:
            logger.error(f"Answering error: {str(e)}")
            st.error("⚠️ Please process documents before asking questions.")
            return None

    def get_summary(self):
        """Generate a summary of all processed documents"""
        if not self.vectorstore:
            return None

        summary_prompt = (
            "Summarize the documents focusing on key insights, outcomes, and trends. "
            "For financial reports: mention revenue, expenses, profit margins, and trends. "
            "For case studies: highlight problem, solution, implementation, and impacts. "
            "Ensure the summary is concise and actionable."
        )
        return self.get_answer(summary_prompt)


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("📊 AI Powered Research Assistant")

    tool = ResearchAssistant()

    # Sidebar input
    with st.sidebar:
        st.header("⚙️ Input Options")
        input_type = st.radio("Choose input type:", ["URLs", "URL File"])

        if input_type == "URLs":
            urls = st.text_area("Enter URLs (one per line)")
            url_list = urls.split("\n") if urls else []
        else:
            uploaded_file = st.file_uploader("Upload file with URLs", type=["txt"])
            url_list = (
                uploaded_file.getvalue().decode().splitlines() if uploaded_file else []
            )

        process_button = st.button("🚀 Process Documents")

    # Document processing
    if process_button and url_list:
        with st.spinner("🔄 Processing documents..."):
            docs = tool.process_urls(url_list)
            if docs and tool.create_embeddings(docs):
                st.success("✅ Processing complete!")

                with st.spinner("📝 Generating summary..."):
                    summary = tool.get_summary()
                    if summary:
                        st.subheader("📌 Document Summary")
                        st.write(summary["answer"])
                        st.subheader("🔗 Sources")
                        st.write(summary["sources"])

    # Query Section
    st.header("💬 Ask a Question")
    query = st.text_input("Type your research question here...")

    if query:
        with st.spinner("🤖 Finding answer..."):
            result = tool.get_answer(query)
            if result:
                st.subheader("📌 Answer")
                st.write(result["answer"])
                st.subheader("🔗 Sources")
                st.write(result["sources"])


if __name__ == "__main__":
    main()
