# app.py
from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
import traceback
import json
from config import PDF_DIR, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_CACHE_DIR, CACHE_ROOT_DIR
from document_processor import DocumentProcessor
from vector_store import ChromaVectorStore
from rag_engine import RAGEngine

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# app.py (添加在導入部分下方)
app = Flask(__name__, 
            static_url_path='/static', 
            static_folder='../static',
            template_folder='../templates')
rag_engine = None
db_info = {}
_initialized = False

@app.route('/')
def index():
    """渲染聊天界面"""
    return render_template('index.html', db_info=db_info)

@app.route('/api/chat', methods=['POST'])
def chat():
    """處理聊天API請求"""
    if not rag_engine:
        return jsonify({
            'status': 'error',
            'message': 'RAG引擎尚未初始化',
            'answer': '系統錯誤：RAG引擎尚未初始化，請重新啟動應用。',
            'docs': []
        })
    
    data = request.json
    query = data.get('query', '')
    
    if not query.strip():
        return jsonify({
            'status': 'error',
            'message': '查詢不能為空',
            'answer': '請輸入問題。',
            'docs': []
        })
    
    try:
        answer, retrieved_docs = rag_engine.process_query(query)
        
        docs_for_display = []
        for i, doc in enumerate(retrieved_docs, 1):
            docs_for_display.append({
                'rank': i,
                'text': doc['text'],
                'source': doc['source'],
            })
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'docs': docs_for_display
        })
    except Exception as e:
        logger.error(f"處理查詢時出錯: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'處理查詢時出錯: {str(e)}',
            'answer': '很抱歉，處理您的問題時遇到了錯誤。請重試或聯繫管理員。',
            'docs': []
        })

def process_documents():
    """
    加載並處理目錄中的文檔，生成文本塊。
    包括:
        - PDF 文件
        - CSV 文件（資料探勘邏輯）
    Returns:
        List[Dict]: 包含文本塊的列表
    """
    logger.info("正在加載並處理文檔...")

    try:
        chunks = DocumentProcessor.process_documents(PDF_DIR, use_inline_images=True)
        if not chunks:
            logger.error("未能從任何文檔中提取任何文本塊")
            return None
        logger.info(f"已成功生成 {len(chunks)} 個文本塊")
        return chunks
    except Exception as e:
        logger.error(f"處理文檔時發生錯誤: {e}", exc_info=True)
        return None

def create_vector_embeddings(vector_store, chunks):
    """
    將文本塊轉換為向量嵌入並存儲。
    Args:
        vector_store (ChromaVectorStore): 向量存儲的對象
        chunks (List[Dict]): 文檔分塊
    Returns:
        int: 成功添加的向量嵌入數量
    """
    logger.info("正在生成向量嵌入並添加到數據庫...")
    try:
        added_count = vector_store.add_documents(chunks)
        if added_count <= 0:
            logger.warning("未能將任何文檔塊添加到向量數據庫")
            return 0
        logger.info(f"向量嵌入處理完成！已成功添加 {added_count} 個文檔塊")
        return added_count
    except Exception as e:
        logger.error(f"添加文檔到向量存儲時發生錯誤: {e}", exc_info=True)
        return 0

def process_incremental_documents(vector_store):
    """
    增量處理文檔:
    1. 檢查是否有新檔案
    2. 檢查文件是否有緩存
    3. 確認緩存資料是否在DB中
    4. 如無緩存，進行chunking和embedding
    """
    logger.info("開始增量處理文檔...")
    
    # 獲取所有文件路徑
    all_files = []
    for file_type in ['.pdf', '.csv']:
        all_files.extend([f for f in os.listdir(PDF_DIR) if f.lower().endswith(file_type)])
    
    if not all_files:
        logger.info("沒有找到任何文檔")
        return []
        
    logger.info(f"找到 {len(all_files)} 個文檔文件")
    
    # 獲取DB中已有的文檔來源
    existing_sources = vector_store.get_all_document_sources()
    logger.info(f"向量數據庫中已有 {len(existing_sources)} 個文檔來源")
    
    # 需要處理的新文檔和已有緩存但未入庫的文檔
    new_chunks = []
    
    for filename in all_files:
        file_path = os.path.join(PDF_DIR, filename)
        
        # 1. 檢查文件是否已在DB中
        if filename in existing_sources:
            logger.info(f"文檔 {filename} 已在數據庫中，跳過處理")
            continue
            
        # 2. 檢查文件是否有chunk緩存
        cache_file = os.path.join(CHUNK_CACHE_DIR, f"{filename}_chunks.json")
        cached_chunks = []
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_chunks = json.load(f)
                logger.info(f"找到文檔 {filename} 的緩存: {len(cached_chunks)} 個chunk")
                
                # 3. 確認緩存資料是否已在DB中 (根據chunk_id檢查)
                chunk_ids = [chunk["chunk_id"] for chunk in cached_chunks]
                existing_chunks = vector_store.check_chunks_exist(chunk_ids)
                
                if not existing_chunks:
                    # 緩存的chunks不在DB中，加入處理列表
                    new_chunks.extend(cached_chunks)
                    logger.info(f"文檔 {filename} 的緩存chunks不在DB中，將使用緩存進行處理")
                else:
                    logger.info(f"文檔 {filename} 的部分或全部chunks已在DB中")
                    
                continue
            except Exception as e:
                logger.warning(f"讀取緩存文件 {cache_file} 失敗: {e}")
                # 緩存讀取失敗，進行正常處理
        
        # 4. 如果沒有緩存或緩存無效，進行正常處理
        logger.info(f"開始處理文檔: {filename}")
        
        try:
            # 根據文件類型選擇處理方法
            if filename.lower().endswith('.pdf'):
                docs = DocumentProcessor.load_pdf_documents_with_inline_images(PDF_DIR, filename_filter=filename)
            elif filename.lower().endswith('.csv'):
                docs = DocumentProcessor.load_csv_documents(PDF_DIR, filename_filter=filename)
            else:
                logger.warning(f"不支持的文件類型: {filename}")
                continue
                
            if docs:
                # 分塊處理
                file_chunks = DocumentProcessor.split_documents(docs)
                
                if file_chunks:
                    # 保存chunk緩存
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(file_chunks, f, ensure_ascii=False, indent=2, 
                                     default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, bool, type(None))) else x)
                        logger.info(f"已保存文檔 {filename} 的chunk緩存: {len(file_chunks)} 個chunk")
                    except Exception as e:
                        logger.warning(f"保存chunk緩存失敗: {e}")
                    
                    # 加入待處理列表
                    new_chunks.extend(file_chunks)
                
        except Exception as e:
            logger.error(f"處理文檔 {filename} 時出錯: {e}", exc_info=True)
    
    logger.info(f"增量處理完成，共發現 {len(new_chunks)} 個新的chunks")
    return new_chunks

def initialize_system():
    """
    初始化RAG系統，包括:
    1. 檢查必要目錄
    2. 初始化向量數據庫連接
    3. 檢查並處理文檔(如果需要)
    4. 初始化RAG引擎
    
    Returns:
        bool: 初始化是否成功
    """
    global rag_engine, db_info, _initialized

    # 防止重複初始化
    if _initialized and rag_engine is not None:
        logger.info("系統已初始化，跳過重複初始化步驟")
        return True
    
    logger.info("RAG Web應用初始化程序")
    
    # 確保必要的目錄存在
    # 確保基本目錄存在
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(CACHE_ROOT_DIR, exist_ok=True)
    
    db_info = {
        'db_dir': CHROMA_DB_DIR,
        'collection_name': CHROMA_COLLECTION_NAME,
        'status': '初始化中...'
    }
    
    try:
        # 初始化向量存儲
        vector_store = ChromaVectorStore(CHROMA_DB_DIR, CHROMA_COLLECTION_NAME)
        
        # 檢查集合是否已存在且有數據
        existing_doc_count = vector_store.count_documents()
        
        if existing_doc_count > 0:
            logger.info(f"找到現有的向量數據庫集合 '{CHROMA_COLLECTION_NAME}'")
            logger.info(f"集合中已存在 {existing_doc_count} 個文檔塊")
            
            # 增加檢查是否有新文檔的邏輯
            logger.info("檢查是否有新增文檔...")
            
            new_chunks = process_incremental_documents(vector_store)
            
            if new_chunks:
                logger.info(f"發現 {len(new_chunks)} 個新文檔塊，準備添加到向量數據庫")
                added_count = create_vector_embeddings(vector_store, new_chunks)
                if added_count > 0:
                    logger.info(f"成功添加 {added_count} 個新文檔塊")
                    db_info['status'] = f'已載入 ({existing_doc_count + added_count} 文檔塊)'
                else:
                    logger.warning("未能成功添加任何新文檔塊")
                    db_info['status'] = f'已載入 ({existing_doc_count} 文檔塊)'
            else:
                logger.info("未發現新文檔或所有文檔已處理")
                db_info['status'] = f'已載入 ({existing_doc_count} 文檔塊)'
        else:
            logger.info(f"未在集合 '{CHROMA_COLLECTION_NAME}' 中找到現有數據，或集合為空")
            logger.info("開始檢查並處理文檔以建立向量數據庫...")
            db_info['status'] = '正在創建...'
            
            # 處理所有文檔
            chunks = process_documents()
            if not chunks:
                db_info['status'] = '錯誤：文檔處理失敗'
                return False
            
            db_info['chunk_count'] = len(chunks)
            
            # 處理向量嵌入
            added_count = create_vector_embeddings(vector_store, chunks)
            if added_count <= 0:
                db_info['status'] = '警告：嵌入生成失敗'
                return False
            
            db_info['status'] = f'已創建 ({added_count} 文檔塊)'
        
        # 初始化RAG引擎
        logger.info("正在初始化RAG引擎...")
        rag_engine = RAGEngine(vector_store)
        logger.info("RAG引擎初始化完成")
        
        logger.info("="*30)
        logger.info("系統初始化完畢，Web應用程序可以開始接收請求")
        logger.info("="*30)
        
        # 標記為已初始化
        _initialized = True
        return True
    
    except Exception as e:
        logger.error(f"初始化系統時發生錯誤: {e}", exc_info=True)
        db_info['status'] = f'錯誤：{str(e)}'
        return False

@app.route('/api/db_info')
def get_db_info():
    """獲取數據庫信息"""
    return jsonify(db_info)

# 全局緩存，保存處理結果
_debug_chunks_cache = {}

@app.route('/api/debug/chunks_with_overlap')
def debug_chunks_with_overlap():
    """診斷視覺化顯示文檔完整分塊結果，包括圖片描述"""
    global _debug_chunks_cache
    import html

    try:
        # 獲取參數
        chunk_size = int(request.args.get('chunk_size', str(CHUNK_SIZE)))
        chunk_overlap = int(request.args.get('overlap', str(CHUNK_OVERLAP)))
        
        # 生成緩存鍵
        cache_key = f"{chunk_size}_{chunk_overlap}_True"
        logger.info(f"準備以參數 chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, use_inline=True 進行分塊可視化")
        
        # 檢查緩存
        if cache_key in _debug_chunks_cache:
            logger.info(f"使用緩存的分塊結果 (緩存鍵: {cache_key})")
            results = _debug_chunks_cache[cache_key]
        else:
            logger.info(f"緩存未命中，開始生成新的分塊結果 (緩存鍵: {cache_key})")
            
            import config
            original_chunk_size = CHUNK_SIZE
            original_chunk_overlap = CHUNK_OVERLAP
            
            # 應用請求的分塊參數
            config.CHUNK_SIZE = chunk_size
            config.CHUNK_OVERLAP = chunk_overlap
            
            try:
                documents = DocumentProcessor.process_documents(PDF_DIR, use_inline_images=True)
            except ImportError:
                documents = DocumentProcessor.process_documents(PDF_DIR, use_inline_images=False)
                logger.warning("未安裝 pdfplumber，無法使用內聯模式，已回退到標準模式")
            
            # 恢復原始配置參數
            config.CHUNK_SIZE = original_chunk_size
            config.CHUNK_OVERLAP = original_chunk_overlap
            
            # 構建結果
            results = []
            # 為每個文檔創建一個結果項
            for doc_source in set(chunk["source"] for chunk in documents):
                doc_chunks = [c for c in documents if c["source"] == doc_source]
                
                # 根據 chunk_id 解析和排序
                doc_chunks.sort(key=lambda c: int(c["chunk_id"].split("_chunk_")[1]))
                
                doc_result = {
                    "filename": doc_source,
                    "total_length": sum(len(c["text"]) for c in doc_chunks),
                    "chunk_count": len(doc_chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunks": []
                }
                
                # 建立塊與前一個塊的關係，以識別重疊
                prev_text = ""
                for i, chunk in enumerate(doc_chunks):
                    chunk_text = chunk["text"]
                    chunk_id = chunk["chunk_id"]
                    contains_image = chunk.get("contains_image", False)
                    
                    # 識別重疊部分
                    overlap_length = 0
                    has_overlap = False
                    
                    if i > 0 and prev_text:
                        prefix_length = min(64, len(chunk_text))
                        prefix = chunk_text[:prefix_length]
                        
                        if prefix in prev_text:
                            overlap_start = prev_text.find(prefix)
                            overlap_text = prev_text[overlap_start:]
                            
                            # 計算有多少字符重疊
                            for j in range(min(len(overlap_text), len(chunk_text))):
                                if overlap_text[j] != chunk_text[j]:
                                    break
                            overlap_length = j
                            has_overlap = overlap_length > 0
                    
                    if has_overlap and overlap_length > 0:
                        # 高亮重疊部分
                        html_chunk = (
                            '<span class="overlap-highlight">' +
                            html.escape(chunk_text[:overlap_length]) +
                            '</span>' +
                            html.escape(chunk_text[overlap_length:])
                        )
                    else:
                        html_chunk = html.escape(chunk_text)
                    
                    # 如果包含圖片描述，添加特殊標記
                    if contains_image:
                        # 識別圖片描述部分並添加特殊樣式
                        # 尋找圖片描述的標記
                        img_markers = ["===== 圖片描述 =====", "[圖片內容:", "[圖片:", "【圖片描述"]
                        
                        for marker in img_markers:
                            if marker in chunk_text:
                                # 替換HTML中的標記
                                styled_marker = f'<span class="image-desc-marker">{html.escape(marker)}</span>'
                                html_chunk = html_chunk.replace(html.escape(marker), styled_marker)
                    
                    # 添加到塊列表
                    doc_result["chunks"].append({
                        "chunk_id": chunk_id,
                        "index": i,
                        "start": 0,  # 不再有絕對位置
                        "end": len(chunk_text),
                        "length": len(chunk_text),
                        "text": chunk_text,
                        "html": html_chunk,
                        "has_overlap": has_overlap,
                        "overlap_length": overlap_length,
                        "contains_image": contains_image
                    })
                    
                    # 更新前一個塊的文本
                    prev_text = chunk_text
                
                results.append(doc_result)
            
            # 存入緩存
            _debug_chunks_cache[cache_key] = results
            logger.info(f"新的分塊結果已生成並緩存 (緩存鍵: {cache_key})")
        
        # 渲染模板
        return render_template('chunks_view.html', results=results)
        
    except Exception as e:
        logger.error(f"在分塊視覺化中出錯: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

if __name__ == '__main__':
    if initialize_system():
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("系統初始化失敗，Web應用未啟動")
        sys.exit(1)