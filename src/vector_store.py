# vector_store.py
import os
import logging
import hashlib
import json
import time
import chromadb
from typing import List, Dict, Optional
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
    AZURE_OPENAI_VERSION,
    CACHE_ROOT_DIR,
    EMBEDDING_CACHE_DIR
)

# 獲取logger
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """向量嵌入生成器，包含緩存功能"""
    
    def __init__(self):
        """初始化Azure OpenAI嵌入客戶端以及緩存系統"""
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_VERSION
        )
        self.deployment_name = AZURE_EMBEDDING_DEPLOYMENT_NAME
        
        # 確保嵌入緩存目錄存在
        os.makedirs(CACHE_ROOT_DIR, exist_ok=True)
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
        
        # 緩存信息
        self.cache_info_path = os.path.join(EMBEDDING_CACHE_DIR, "cache_info.json")
        self.cache_info = self._load_cache_info()
        
        logger.info(f"嵌入生成器初始化完成，使用部署: {self.deployment_name}")
        logger.info(f"嵌入緩存目錄: {EMBEDDING_CACHE_DIR}")
        logger.info(f"載入了 {len(self.cache_info)} 個緩存項")

    def _get_text_hash(self, text: str) -> str:
        """計算文本的雜湊值作為緩存鍵"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, text_hash: str) -> Optional[List[float]]:
        """從緩存中獲取向量嵌入"""
        if text_hash in self.cache_info:
            cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{text_hash}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"讀取緩存文件 {text_hash} 時出錯: {e}")
        return None
    
    def _load_cache_info(self) -> Dict:
        """載入緩存信息"""
        if os.path.exists(self.cache_info_path):
            try:
                with open(self.cache_info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"載入緩存信息時出錯: {e}，將創建新的緩存")
        return {}
    
    def _save_to_cache(self, text_hash: str, embedding: List[float]):
        """將向量嵌入保存到緩存"""
        cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{text_hash}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(embedding, f)
            
            # 更新緩存信息
            self.cache_info[text_hash] = {
                "timestamp": time.time(),
                "size": len(embedding)
            }
            self._save_cache_info()
        except Exception as e:
            logger.warning(f"保存緩存文件 {text_hash} 時出錯: {e}")

    def _save_cache_info(self):
        """保存緩存信息"""
        try:
            with open(self.cache_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存緩存信息時出錯: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """為一組文本生成嵌入向量，優先從緩存中獲取"""
        if not texts:
            logger.warning("嘗試為空文本列表生成嵌入")
            return []
            
        try:
            logger.info(f"處理 {len(texts)} 個文本的嵌入向量...")
            
            # 先檢查哪些文本可以從緩存獲取
            results = []
            texts_to_generate = []
            hashes = []
            cache_hits = 0
            
            for text in texts:
                text_hash = self._get_text_hash(text)
                cached_embedding = self._get_from_cache(text_hash)
                
                if cached_embedding:
                    results.append(cached_embedding)
                    cache_hits += 1
                else:
                    results.append(None)  # 先填入None，稍後替換
                    texts_to_generate.append(text)
                    hashes.append(text_hash)
            
            logger.info(f"緩存命中率: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.2f}%)")
            
            # 如果所有文本都從緩存獲取了，直接返回
            if not texts_to_generate:
                logger.info(f"所有 {len(texts)} 個嵌入向量均從緩存獲取")
                return results
            
            # 處理未緩存的文本，使用批次處理和錯誤重試
            logger.info(f"需要生成 {len(texts_to_generate)} 個新嵌入向量")
            
            batch_size = 50  # 每次處理50個文本
            for i in range(0, len(texts_to_generate), batch_size):
                batch_texts = texts_to_generate[i:i+batch_size]
                batch_hashes = hashes[i:i+batch_size]
                logger.info(f"處理批次 {i//batch_size + 1}/{(len(texts_to_generate)-1)//batch_size + 1}，包含 {len(batch_texts)} 個文本")
                
                # 嘗試處理當前批次，如果遇到限流問題則增加等待時間
                max_retries = 5
                retry_count = 0
                retry_delay = 5  # 初始延遲5秒
                
                while retry_count < max_retries:
                    try:
                        response = self.client.embeddings.create(
                            input=batch_texts,
                            model=self.deployment_name
                        )
                        # 從回應獲取嵌入向量
                        batch_embeddings = [item.embedding for item in response.data]
                        
                        # 更新結果並保存到緩存
                        for j, (text_hash, embedding) in enumerate(zip(batch_hashes, batch_embeddings)):
                            # 找回原始索引位置
                            original_idx = texts.index(texts_to_generate[i+j])
                            results[original_idx] = embedding
                            
                            # 保存到緩存
                            self._save_to_cache(text_hash, embedding)
                        
                        # 成功處理後等待1秒避免過快發送下一批請求
                        time.sleep(1)
                        break
                        
                    except Exception as e:
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            retry_count += 1
                            logger.warning(f"遇到API限流，第{retry_count}次重試，等待{retry_delay}秒...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指數退避，每次失敗後翻倍等待時間
                        else:
                            logger.error(f"生成嵌入時出錯: {str(e)}", exc_info=True)
                            # 為這批次的所有項設置為None
                            for j in range(len(batch_texts)):
                                original_idx = texts.index(texts_to_generate[i+j])
                                results[original_idx] = None
                            break
                
                if retry_count >= max_retries:
                    logger.error(f"批次處理重試次數已超過上限，跳過此批次")
                    for j in range(len(batch_texts)):
                        original_idx = texts.index(texts_to_generate[i+j])
                        results[original_idx] = None
            
            # 報告最終結果
            successful_count = len([r for r in results if r is not None])
            logger.info(f"成功處理 {successful_count}/{len(texts)} 個嵌入向量")
            
            return results
            
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
        """將文檔塊添加到向量存儲，支持批次處理"""
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

        # 使用批次處理添加文檔
        max_batch_size = 1000  # 設置一個安全的批次大小，遠低於限制
        total_added = 0
        
        try:
            logger.info(f"開始批次添加 {len(valid_data)} 個文檔塊到集合 '{self.collection_name}'...")
            
            # 分批處理
            for i in range(0, len(valid_data), max_batch_size):
                batch = valid_data[i:i+max_batch_size]
                batch_size = len(batch)
                
                logger.info(f"處理批次 {i//max_batch_size + 1}/{(len(valid_data)-1)//max_batch_size + 1}: {batch_size} 個文檔塊")
                
                # 準備批量添加的數據
                add_ids = [item["id"] for item in batch]
                add_embeddings = [item["embedding"] for item in batch]
                add_documents = [item["text"] for item in batch]
                add_metadatas = [item["metadata"] for item in batch]
                
                # 添加當前批次
                self.collection.add(
                    ids=add_ids,
                    embeddings=add_embeddings,
                    documents=add_documents,
                    metadatas=add_metadatas
                )
                
                total_added += batch_size
                logger.info(f"成功添加批次 {i//max_batch_size + 1}, 總計已添加: {total_added}/{len(valid_data)}")
            
            logger.info(f"成功添加/更新所有 {total_added} 個文檔塊。")
            return total_added
            
        except Exception as e:
            logger.error(f"向集合 '{self.collection_name}' 添加文檔時出錯: {e}", exc_info=True)
            # 如果部分批次已成功添加，返回已添加的數量
            if total_added > 0:
                logger.warning(f"部分添加成功：已添加 {total_added} 個文檔塊")
            return total_added
    
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
    
    def check_chunks_exist(self, chunk_ids: List[str]) -> List[str]:
        """檢查指定chunk_id是否已存在於數據庫中"""
        try:
            existing_ids = []
            # 分批檢查以避免請求過大
            batch_size = 100
            for i in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[i:i+batch_size]
                # ChromaDB的get方法會返回找到的ID
                results = self.collection.get(ids=batch_ids, include=[])
                if results and results.get("ids"):
                    existing_ids.extend(results["ids"])
            
            return existing_ids
        except Exception as e:
            logger.error(f"檢查chunks存在性時出錯: {e}", exc_info=True)
            return []
    
    def get_all_document_sources(self) -> List[str]:
        """獲取向量存儲中所有文檔的來源名稱"""
        try:
            result = self.collection.get(include=["metadatas"])
            if result and result.get("metadatas"):
                # 從metadata中提取來源檔名
                sources = set()
                for metadata in result["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])
                return list(sources)
            return []
        except Exception as e:
            logger.error(f"獲取文檔來源時出錯: {e}", exc_info=True)
            return []