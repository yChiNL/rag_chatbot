import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI配置
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_VERSION_DEPLOYMENT_NAME = os.getenv("AZURE_VERSION_DEPLOYMENT_NAME")
AZURE_VISION_MINI_DEPLOYMENT_NAME = os.getenv("AZURE_VISION_MINI_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")

# 文件路徑
PDF_DIR = os.path.join(os.path.dirname(__file__), "../data/documents")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
IMAGE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/_image_cache")

# 文本處理配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
CHROMA_COLLECTION_NAME = "document_chunks"

# RAG配置
TOP_K_RESULTS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 800