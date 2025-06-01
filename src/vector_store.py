import os
import logging
import chromadb
from typing import List, Dict, Optional
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
    AZURE_OPENAI_VERSION
)

# 獲取logger
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """向量嵌入生成器"""
    
    def __init__(self):
        """初始化Azure OpenAI嵌入客戶端"""
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_VERSION
        )
        self.deployment_name = AZURE_EMBEDDING_DEPLOYMENT_NAME
        logger.info(f"嵌入生成器初始化完成，使用部署: {self.deployment_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """為一組文本生成嵌入向量"""
        if not texts:
            logger.warning("嘗試為空文本列表生成嵌入")
            return []
            
        try:
            logger.info(f"正在為 {len(texts)} 個文本生成嵌入...")
            response = self.client.embeddings.create(
                input=texts,
                model=self.deployment_name
            )
            # 從回應獲取嵌入向量
            embeddings = [item.embedding for item in response.data]
            logger.info(f"成功生成 {len(embeddings)} 個嵌入向量")
            return embeddings
            
        except Exception as e:
            logger.error(f"生成嵌入時出錯: {str(e)}", exc_info=True)
            
            return [None] * len(texts)

class ChromaVectorStore:
    """Chroma向量數據庫客戶端"""
    
    def __init__(self, db_directory: str, collection_name: str): 
        """初始化ChromaDB客戶端"""
        os.makedirs(db_directory, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=db_directory)
            self.collection_name = collection_name
            self.embedding_generator = EmbeddingGenerator()
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}   
            )
            logger.info(f"成功連接到 ChromaDB 集合: '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"連接或創建 ChromaDB 集合 '{collection_name}' 時出錯: {e}", exc_info=True)
            raise
    
    def count_documents(self) -> int:
        """返回集合中文檔塊的數量"""
        try:
            count = self.collection.count()
            logger.info(f"集合 '{self.collection_name}' 中有 {count} 個文檔塊")
            return count
        except Exception as e:
            logger.error(f"查詢集合 '{self.collection_name}' 文檔數量時出錯: {e}", exc_info=True)
            return 0
        
    def add_documents(self, chunks: List[Dict]) -> int:
        """將文檔塊添加到向量存儲"""
        if not chunks:
            logger.warning("沒有文檔塊需要添加到向量存儲。")
            return 0

        # 提取必要的字段
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        
        # 建立元數據
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source": chunk["source"],
                "path": chunk.get("path", ""),
                "original_chunk_id": chunk["chunk_id"],  
                "contains_image": chunk.get("contains_image", False)
            }
            metadatas.append(metadata)
        
        logger.info(f"正在為 {len(texts)} 個文本塊生成嵌入向量...")
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # 檢查每個嵌入是否生成成功
        valid_data = []
        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                valid_data.append({
                    "id": ids[i],
                    "text": texts[i],
                    "metadata": metadatas[i],
                    "embedding": embedding
                })
            else:
                logger.warning(f"文檔塊 '{ids[i]}' 未能生成嵌入，將被跳過。")

        if not valid_data:
            logger.error("沒有任何文檔塊成功生成嵌入，無法添加到數據庫。")
            return 0

        try:
            logger.info(f"正在將 {len(valid_data)} 個文檔塊添加到集合 '{self.collection_name}'...")
            
            # 準備批量添加的數據
            add_ids = [item["id"] for item in valid_data]
            add_embeddings = [item["embedding"] for item in valid_data]
            add_documents = [item["text"] for item in valid_data]
            add_metadatas = [item["metadata"] for item in valid_data]
            
            self.collection.add(
                ids=add_ids,
                embeddings=add_embeddings,
                documents=add_documents,
                metadatas=add_metadatas
            )
            
            logger.info(f"成功添加/更新 {len(valid_data)} 個文檔塊。")
            return len(valid_data)
            
        except Exception as e:
            logger.error(f"向集合 '{self.collection_name}' 添加文檔時出錯: {e}", exc_info=True)
            return 0
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        使用語義相似度在向量數據庫中搜索與查詢最相關的文檔
        
        Args:
            query: 用戶查詢文本
            limit: 返回結果數量上限
            
        Returns:
            List[Dict]: 包含相關文檔信息的列表，每個文檔包含文本、來源、相似度等字段
            若發生錯誤或無結果則返回空列表
        """
        if self.count_documents() == 0:
            logger.warning(f"向量數據庫集合 '{self.collection_name}' 為空，無法執行搜索。")
            return []

        try:
            logger.info(f"為查詢生成嵌入: '{query}'")
            query_embedding_list = self.embedding_generator.generate_embeddings([query])
            
            if not query_embedding_list or query_embedding_list[0] is None:
                logger.error("未能為查詢生成嵌入向量。")
                return []
                
            query_embedding = query_embedding_list[0]

            logger.info(f"正在向量數據庫中搜索與查詢最相關的 {limit} 個文檔...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            logger.error(f"在集合 '{self.collection_name}' 中搜索時出錯: {e}", exc_info=True)
            return []

        documents = []
        if results and results.get("documents") and results["documents"][0] is not None:
            ids = results.get("ids", [["unknown_chunk_" + str(i) for i in range(len(results["documents"][0]))]])

            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}
                doc_id = ids[0][i] if ids and len(ids) > 0 and len(ids[0]) > i else f"unknown_chunk_{i}" 
                distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else 0
                
                chunk_id = metadata.get("original_chunk_id", doc_id)
                
                documents.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "source": metadata.get("source", "未知來源"),
                    "similarity_score": 1 - distance,
                    "metadata": metadata 
                })
                
            logger.info(f"搜索完成，找到 {len(documents)} 個相關文檔塊")
        else:
            logger.warning("搜索未返回任何文檔，或結果格式不符合預期。")

        return documents