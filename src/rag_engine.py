import logging
import yaml
import os
from typing import Dict, List, Tuple
from openai import AzureOpenAI
from vector_store import ChromaVectorStore
from config import (
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_VERSION_DEPLOYMENT_NAME,
    AZURE_OPENAI_VERSION,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K_RESULTS
)

# 獲取logger
logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG引擎 - 整合檢索、提示構建和LLM調用"""
    
    def __init__(self, vector_store: ChromaVectorStore):
        """初始化RAG引擎"""
        self.vector_store = vector_store
        try:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_VERSION
            )
            self.deployment_name = AZURE_VERSION_DEPLOYMENT_NAME
            self.top_k = TOP_K_RESULTS
            self.temperature = TEMPERATURE
            self.max_tokens = MAX_TOKENS
            
            logger.info(f"RAG引擎初始化完成，使用部署: {self.deployment_name}，溫度: {self.temperature}")
            logger.info(f"檢索設置: top_k={self.top_k}, max_tokens={self.max_tokens}")
        except Exception as e:
            logger.error(f"初始化RAG引擎時出錯: {e}", exc_info=True)
            raise
    
    def retrieve_documents(self, query: str) -> List[Dict]:
        """檢索與查詢相關的文檔"""
        logger.info(f"檢索與查詢相關的文檔: '{query}'")
        try:
            docs = self.vector_store.search(query, self.top_k)
            if docs:
                logger.info(f"檢索到 {len(docs)} 個相關文檔")

                for i, doc in enumerate(docs):
                    score = doc.get('similarity_score', 'N/A')
                    source = doc.get('source', 'unknown')
                    logger.debug(f"檢索結果 #{i+1}: 相似度 {score:.4f}")
            else:
                logger.warning("未檢索到任何相關文檔")
                
            return docs
        except Exception as e:
            logger.error(f"在檢索相關文檔時發生錯誤: {e}", exc_info=True)
            return []
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> List[Dict]:
        try:
            sorted_docs = sorted(
                retrieved_docs, 
                key=lambda x: x.get('similarity_score', 0), 
                reverse=True
            )
            
            context_parts = []
            docs_included = 0
            max_docs = min(self.top_k, len(sorted_docs))
            
            # 跟踪是否包含了圖片描述
            has_image_desc = False
            
            for doc in sorted_docs[:max_docs]:
                doc_text = doc['text']
                source = doc.get('source', '未知來源')
                contains_image = "圖片描述" in doc_text
                
                if contains_image:
                    has_image_desc = True
                
                doc_context = f"文檔內容: {doc_text}\n來源: {source}"
                if contains_image:
                    doc_context = f"[包含圖片描述] {doc_context}" 
                    
                context_parts.append(doc_context)
                docs_included += 1
            
            # 將所有文檔片段組合成完整上下文
            context = "\n\n".join(context_parts)

            # 系統提示的基本部分
            with open(os.path.join(os.path.dirname(__file__), "prompts.yaml"), 'r', encoding='utf-8') as f:
                prompts_yaml = yaml.safe_load(f.read())
            system_prompt = prompts_yaml.get("rag_system").format(context=context)
            
            if has_image_desc:
                logger.info(f"最終提示包含 {docs_included} 個文檔，其中包含圖片描述")
            else:
                logger.info(f"最終提示包含 {docs_included} 個文檔，不含圖片描述")
            
            # 構建完整提示消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            return messages
        except Exception as e:
            logger.error(f"構建提示時出錯: {e}", exc_info=True)
            # 返回一個基本提示，避免完全失敗
            return [
                {"role": "system", "content": "你是一個智能助手，請根據你的知識回答問題。如果不確定，請誠實說明。"},
                {"role": "user", "content": query}
            ]

    
    def generate_response(self, messages: List[Dict]) -> str:
        """生成回應"""
        try:
            logger.info(f"使用 {self.deployment_name} 生成回應，溫度: {self.temperature}, 最大token: {self.max_tokens}")
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            return response_content
            
        except Exception as e:
            logger.error(f"生成回應時出錯: {e}", exc_info=True)
            return "很抱歉，生成回應時發生錯誤。請稍後再試或聯繫系統管理員。"
    
    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """處理用戶查詢的主要方法"""
        logger.info(f"處理用戶查詢: '{query}'")
        
        try:
            # 檢索相關文檔
            retrieved_docs = self.retrieve_documents(query)
            
            if not retrieved_docs:
                logger.warning("未找到相關文檔，將使用模型的一般知識回答")
                # 如果沒有檢索到文檔，使用一般提示
                messages = [
                    {"role": "system", "content": "你是一個智能助手。請回答用戶的問題，如果不確定答案，請誠實說明。"},
                    {"role": "user", "content": query}
                ]
            else:
                # 構建RAG提示
                messages = self.build_prompt(query, retrieved_docs)
            
            # 生成回應
            response = self.generate_response(messages)
            logger.info("成功生成回應")
            
            return response, retrieved_docs
            
        except Exception as e:
            logger.error(f"處理查詢過程中發生錯誤: {e}", exc_info=True)
            error_message = "很抱歉，處理您的問題時遇到了技術問題。請稍後再試。"
            return error_message, []