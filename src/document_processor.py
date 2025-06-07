# document_processor.py
import os
import base64
import logging
import yaml
import pypdf
import io
import pdfplumber
import re
from Preprocessing import DataMiningProcessor
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from openai import AzureOpenAI
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_VERSION,
    AZURE_VISION_MINI_DEPLOYMENT_NAME,
    IMAGE_CACHE_DIR
)


# 獲取logger
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文檔處理類 - 負責PDF加載、解析、圖片描述和分塊"""

    @staticmethod
    def _remove_page_markers(text: str) -> str:
        """使用正則表達式移除或標準化頁碼標記"""
        if not text:
            return ""
        cleaned_text = re.sub(r"^\s*===== [頁面碼]+[:：\s]*\d+[:：\s]*=====*\s*$", "", text, flags=re.MULTILINE)

        return cleaned_text.strip()

    @staticmethod
    def get_image_description(image, client, source_info="") -> str:
        """使用GPT-4o獲取圖片描述，包含來源信息"""
        try:
            # 系統提示的基本部分
            with open(os.path.join(os.path.dirname(__file__), "prompts.yaml"), 'r', encoding='utf-8') as f:
                prompts_yaml = yaml.safe_load(f)
                image_description_prompt = prompts_yaml.get("image_description")

            # 將PIL圖像轉換為bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # 使用GPT-4o-mini生成描述
            response = client.chat.completions.create(
                model=AZURE_VISION_MINI_DEPLOYMENT_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": image_description_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"詳細描述這張圖片中的內容：{source_info}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"}}
                        ]
                    }
                ],
                max_tokens=800,
            )
            description = response.choices[0].message.content

            # 在描述前添加來源信息（如果未在描述中包含）
            if source_info and not description.startswith(f"[來源:") and not description.startswith(f"來源:"):
                description = f"[來源: {source_info}]\n{description}"

            return description
        except Exception as e:
            logger.error(f"生成圖片描述時出錯: {e}", exc_info=True)
            return f"[圖片描述失敗 - 來源: {source_info}]"

    @staticmethod
    def is_page_mostly_image(page_text: str) -> bool:
        """判斷頁面是否主要為圖片（文字內容少）"""
        return len(page_text.strip()) < 100

    @staticmethod
    def load_pdf_documents(directory: str) -> List[Dict]:
        """加載所有PDF文件，並為圖片頁面生成描述"""
        documents = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"在目錄 '{directory}' 中未找到任何PDF文檔。")
            return []

        logger.info(f"在 '{directory}' 中找到 {len(pdf_files)} 個PDF文件")

        try:
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_VERSION
            )
        except Exception as e:
            logger.error(f"初始化OpenAI客戶端失敗: {e}", exc_info=True)
            client = None

        cache_dir = IMAGE_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        for filename in pdf_files:
            file_path = os.path.join(directory, filename)
            try:
                logger.info(f"開始處理PDF: {filename}")
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    text_content_with_markers = ""
                    pages_needing_image_desc = []

                    for page_num in range(len(pdf_reader.pages)):
                        page_text = pdf_reader.pages[page_num].extract_text() or ""
                        text_content_with_markers += f"\n===== 頁面 {page_num+1} =====\n{page_text}\n"

                        if DocumentProcessor.is_page_mostly_image(page_text) and client:
                            pages_needing_image_desc.append(page_num)
                            logger.info(f"頁面 {page_num+1} 可能包含重要圖片，將生成描述")

                    if pages_needing_image_desc and client:
                        conda_prefix = os.environ.get('CONDA_PREFIX')
                        poppler_path = None
                        if conda_prefix:
                            poppler_path = os.path.join(conda_prefix, 'bin')

                        try:
                            images = convert_from_path(file_path, poppler_path=poppler_path)
                            for page_num_desc in pages_needing_image_desc:
                                if page_num_desc < len(images):
                                    image_id = f"{filename.replace('.pdf', '')}_page{page_num_desc+1}"
                                    desc_file = os.path.join(cache_dir, f"{image_id}.txt")
                                    source_info = f"{filename} (頁面 {page_num_desc+1})"

                                    if os.path.exists(desc_file):
                                        with open(desc_file, 'r', encoding='utf-8') as cache:
                                            image_desc = cache.read()
                                        logger.info(f"使用緩存的圖片描述: {source_info}")
                                    else:
                                        logger.info(f"生成圖片描述: {source_info}")
                                        image_desc = DocumentProcessor.get_image_description(
                                            images[page_num_desc], client, source_info=source_info
                                        )
                                        try:
                                            with open(desc_file, 'w', encoding='utf-8') as cache:
                                                cache.write(image_desc)
                                        except Exception as ce:
                                            logger.warning(f"無法保存圖片描述緩存: {ce}")

                                    text_content_with_markers += f"""
===== 圖片描述 =====
來源文件: {filename}
頁碼: {page_num_desc+1}
圖片ID: {image_id}
描述內容:
{image_desc}
===== 圖片描述結束 =====

"""
                        except Exception as e_img:
                            error_msg = f"處理圖片時出錯: {e_img}"
                            logger.error(error_msg, exc_info=True)
                            text_content_with_markers += f"\n[{error_msg}]\n"

                # 在添加文檔到結果之前，清理文本
                cleaned_text_content = DocumentProcessor._remove_page_markers(text_content_with_markers)

                documents.append({
                    "filename": filename,
                    "path": file_path,
                    "text": cleaned_text_content
                })
                logger.info(f"成功處理PDF: {filename}")

            except Exception as e:
                logger.error(f"處理文件 {filename} 時出錯: {str(e)}", exc_info=True)
        return documents

    @staticmethod
    def load_pdf_documents_with_inline_images(directory: str) -> List[Dict]:
        documents = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"在目錄 '{directory}' 中未找到任何PDF文檔。")
            return []

        logger.info(f"在 '{directory}' 中找到 {len(pdf_files)} 個PDF文件")

        try:
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_VERSION
            )
        except Exception as e:
            logger.error(f"初始化OpenAI客戶端失敗: {e}", exc_info=True)
            client = None

        cache_dir = IMAGE_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        conda_prefix = os.environ.get('CONDA_PREFIX')
        poppler_path = os.path.join(conda_prefix, 'bin') if conda_prefix else None

        for filename in pdf_files:
            file_path = os.path.join(directory, filename)
            try:
                logger.info(f"開始處理PDF: {filename}")
                with pdfplumber.open(file_path) as pdf:
                    full_text_with_markers = ""

                    page_images_for_pdf = []
                    try:
                        page_images_for_pdf = convert_from_path(file_path, poppler_path=poppler_path)
                    except Exception as e_conv:
                        logger.error(f"無法轉換PDF {filename} 為圖像: {e_conv}", exc_info=True)

                    for page_num, page in enumerate(pdf.pages):
                        logger.info(f"處理 {filename} 第 {page_num+1} 頁 ")
                        page_text_from_pdf = page.extract_text() or ""
                        current_page_content = "" 

                        page_has_images_flag = bool(page.images) 

                        if page_has_images_flag or DocumentProcessor.is_page_mostly_image(page_text_from_pdf):
                            if page_num < len(page_images_for_pdf) and client:
                                image_id = f"{filename.replace('.pdf', '')}_page{page_num+1}"
                                desc_file = os.path.join(cache_dir, f"{image_id}.txt")
                                source_info = f"{filename} (頁面 {page_num+1})"
                                image_desc = ""

                                if os.path.exists(desc_file):
                                    with open(desc_file, 'r', encoding='utf-8') as f:
                                        image_desc = f.read()
                                    logger.info(f"使用緩存的圖片描述: {image_id}")
                                else:
                                    logger.info(f"生成圖片描述: {source_info}")
                                    image_desc = DocumentProcessor.get_image_description(
                                        page_images_for_pdf[page_num], client, source_info=source_info
                                    )
                                    with open(desc_file, 'w', encoding='utf-8') as f:
                                        f.write(image_desc)

                                formatted_desc = f"\n【圖片描述 (頁 {page_num+1})】\n{image_desc}\n"
                                if not page_text_from_pdf.strip():
                                    current_page_content = formatted_desc
                                else:
                                    current_page_content = page_text_from_pdf
                                    current_page_content += "\n" + formatted_desc
                            else:
                                current_page_content = page_text_from_pdf
                        else:
                            current_page_content = page_text_from_pdf

                        full_text_with_markers += f"\n===== 頁面 {page_num+1} =====\n{current_page_content}\n"

                    cleaned_full_text = DocumentProcessor._remove_page_markers(full_text_with_markers)

                    documents.append({
                        "filename": filename,
                        "path": file_path,
                        "text": cleaned_full_text
                    })
                    logger.info(f"成功處理PDF: {filename} ")
            except Exception as e:
                logger.error(f"處理文件 {filename}  時出錯: {str(e)}", exc_info=True)
        return documents

    @staticmethod
    def load_csv_documents(directory: str) -> List[Dict]:
        """加載所有 CSV 檔案並進行資料前處理"""
        documents = []
        csv_files = [f for f in os.listdir(directory) if f.lower().endswith('.csv')]

        if not csv_files:
            logger.warning(f"在目錄 '{directory}' 中未找到任何 CSV 檔案。")
            return []

        logger.info(f"在 '{directory}' 中找到 {len(csv_files)} 個 CSV 檔案")

        for filename in csv_files:
            file_path = os.path.join(directory, filename)
            try:
                logger.info(f"開始處理 CSV 檔案: {filename}")
                processed_data = DataMiningProcessor.preprocess_csv(file_path) # 使用資料探勘進行前處理
                documents.append({
                    "filename": filename,
                    "path": file_path,
                    "text": processed_data,
                    "is_csv": True
                })
                logger.info(f"成功處理 CSV 文件: {filename}")

            except Exception as e:
                logger.error(f"處理 CSV 文件 {filename} 時發生錯誤: {str(e)}", exc_info=True)
        return documents
    
    @staticmethod
    def split_documents(documents: List[Dict], chunk_size: int = CHUNK_SIZE, 
                    chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
        """將文檔切分為小塊"""
        if not documents:
            logger.warning("未提供文檔供分塊處理。")
            return []

        chunks = []
        for doc in documents:
            # 針對CSV文件應用固定的chunking參數
            if doc.get("is_csv", False):
                csv_chunk_size = 1024  # 固定CSV的chunk_size為1024
                csv_chunk_overlap = 256  # 固定CSV的chunk_overlap為256
                logger.info(f"處理CSV文件: {doc['filename']} 使用特定分塊設置: chunk_size={csv_chunk_size}, chunk_overlap={csv_chunk_overlap}")
                
                # 處理CSV文件的特殊分塊邏輯 - 根據字符長度而非行數進行分塊
                lines = doc["text"].split('\n')
                lines = [line for line in lines if line.strip()]  # 過濾空行
                
                current_chunk_text = ""
                line_index = 0
                
                while line_index < len(lines):
                    line = lines[line_index]
                    
                    # 如果當前行加入後會超過chunk_size，則創建新chunk
                    if len(current_chunk_text) + len(line) > csv_chunk_size and current_chunk_text:
                        chunk_id = f"{doc['filename']}_chunk_{len(chunks)}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": current_chunk_text,
                            "source": doc["filename"],
                            "path": doc["path"],
                            "contains_image": False  # CSV不包含圖片
                        })
                        
                        # 計算重疊部分
                        # 找到重疊部分的起始行
                        overlap_text = ""
                        overlap_lines = []
                        current_length = 0
                        
                        # 從當前chunk末尾反向找出足夠的行以滿足重疊需求
                        temp_lines = current_chunk_text.split('\n')
                        for i in range(len(temp_lines) - 1, -1, -1):
                            line_length = len(temp_lines[i]) + 1  # +1 為換行符
                            if current_length + line_length <= csv_chunk_overlap:
                                overlap_lines.insert(0, temp_lines[i])
                                current_length += line_length
                            else:
                                break
                        
                        current_chunk_text = '\n'.join(overlap_lines) if overlap_lines else ""
                    
                    # 將當前行加入到chunk
                    if current_chunk_text and not current_chunk_text.endswith('\n'):
                        current_chunk_text += '\n'
                    current_chunk_text += line
                    line_index += 1
                    
                # 確保最後一個chunk也被處理
                if current_chunk_text:
                    chunk_id = f"{doc['filename']}_chunk_{len(chunks)}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk_text,
                        "source": doc["filename"],
                        "path": doc["path"],
                        "contains_image": False
                    })
                
                logger.info(f"CSV文件 {doc['filename']} 分為 {len(chunks) - len([c for c in chunks if c['source'] != doc['filename']])} 個文本塊")
                
            else:
                # 對於非CSV文件(如PDF)，使用config中的參數
                logger.info(f"處理非CSV文件: {doc['filename']} 使用配置分塊設置: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", "。", "，", "！", "？", " ", ""]
                )
                
                texts = text_splitter.split_text(doc["text"])
                logger.info(f"非CSV文件 '{doc['filename']}' 分為 {len(texts)} 個文本塊")
                
                for i, text_chunk in enumerate(texts):
                    has_image_desc = "圖片描述" in text_chunk or "[圖片內容:" in text_chunk or "[圖片:" in text_chunk or "【圖片描述" in text_chunk
                    
                    chunk_id = f"{doc['filename']}_chunk_{i}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": text_chunk,
                        "source": doc["filename"],
                        "path": doc["path"],
                        "contains_image": has_image_desc
                    })

        image_chunks = sum(1 for c in chunks if c.get("contains_image", False))
        if image_chunks > 0:
            logger.info(f"共有 {image_chunks} 個文本塊包含圖片描述")

        logger.info(f"文檔分塊完成，總共生成 {len(chunks)} 個文本塊")
        return chunks

    @classmethod
    def process_documents(cls, directory: str, use_inline_images: bool = True) -> List[Dict]:
        """處理PDF和CSV文檔的主入口方法
    
        參數:
            directory: 文檔目錄
            use_inline_images: 是否將圖片描述插入到原始位置，需要安裝pdfplumber
        """
        logger.info(f"開始處理目錄 '{directory}' 中的文檔...")

        if use_inline_images:
            logger.info("使用內聯圖片描述模式")
            try:
                documents = cls.load_pdf_documents_with_inline_images(directory)
            except ImportError:
                logger.warning("未安裝pdfplumber，回退到標準處理模式")
                documents = cls.load_pdf_documents(directory)
        else:
            logger.info("使用標準處理模式")
            documents = cls.load_pdf_documents(directory)

        if not documents:
            logger.warning("沒有找到任何PDF文檔或所有文檔解析失敗。")
            return []

        chunks = cls.split_documents(documents)
        logger.info(f"文檔處理完成，總共生成了 {len(chunks)} 個文本塊")
        return chunks