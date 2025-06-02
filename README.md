# RAG 聊天機器人

## 專案簡介

RAG 聊天機器人是一個基於檢索增強生成（Retrieval-Augmented Generation, RAG）技術的智慧問答系統，可根據上傳的 PDF 文件內容準確回答用戶提問。系統支援文字與圖片內容處理，藉由向量數據庫實現高效的語義檢索，並透過大型語言模型生成自然、相關的回答。

---

## 功能特點

- **PDF 文檔處理**：自動載入、解析及分塊 PDF 文檔。
- **圖像內容識別**：對 PDF 中的圖表和圖片進行智慧描述。
- **語義檢索**：使用向量資料庫進行相關內容的精準檢索。
- **智慧問答**：基於檢索結果生成準確、相關的回答。
- **互動式介面**：提供網頁對話介面。
- **分塊視覺化**：支援文檔分塊結果的視覺化展示與調試。

---

## 技術架構

- **前端**：HTML、CSS、JavaScript。
- **後端**：Flask (Python)。
- **向量資料庫**：ChromaDB。
- **文本嵌入**：Azure OpenAI Embeddings。
- **語言模型**：Azure OpenAI 4o、Azure OpenAI 4o-mini。
- **PDF 處理**：PyPDF、PDFPlumber、PDF2Image。

---

## 安裝步驟

1. **複製儲存庫**

    ```bash
    git clone git@github.com:yChiNL/rag_chatbot.git
    ```

2. **建立虛擬環境**

    ```bash
    conda create -n rag_env python=3.12
    conda activate rag_env
    ```

3. **安裝依賴套件**

    ```bash
    pip install -r requirements.txt
    ```

4. **安裝 Poppler**

    因Windows與Mac安裝方法不同，因此無列於`requirements.txt`中 (Release Version>=21.10.0)
    
    - macOS：
    使用 Homebrew 安裝：
    
    ```bash
    brew install poppler
    ```

    - Windows：
    1. 下載 Poppler 編譯包:

        前往 [Poppler Windows Release](https://github.com/oschwartz10612/poppler-windows/releases) 下載最新版本的 poppler-xxx_x64.zip 文件。

    2. 解壓文件:

        將文件解壓到指定目錄（例如：`C:\Program Files\poppler`）。

    3. 配置系統 PATH:

        添加 Poppler 的 bin 文件夾到環境變數：

        - 打開「系統環境變量」設置，編輯 PATH。
        - 添加 Poppler 的 bin 路徑，例如：C:\Program Files\poppler\bin。
        - 保存並重啟終端。

5. **配置 Azure OpenAI 環境變數**

    建立 `.env` 文件並添加以下內容：

    ```plaintext
    AZURE_OPENAI_API_KEY=your_api_key
    AZURE_OPENAI_ENDPOINT=your_endpoint
    AZURE_VERSION_DEPLOYMENT_NAME=your_deployment_name
    AZURE_VISION_MINI_DEPLOYMENT_NAME=your_vision_deployment_name
    AZURE_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name
    AZURE_OPENAI_VERSION=your_api_version
    ```

6. **放置 PDF 文檔**

    將需要查詢的 PDF 文檔放入 `data/documents/` 目錄中。

---

## 使用說明

### 啟動應用

```bash
cd src
python app.py
```

應用程式將在 http://localhost:5001 上啟動。

### 使用介面
開啟瀏覽器訪問 http://localhost:5001。
等待系統初始化完成 (首次啟動時會處理 PDF 文件並建立向量索引)。
在聊天框中輸入問題，點擊「發送」或按 Enter 鍵提交。
查看系統回覆及相關文檔的來源。

### 分塊視覺化工具
訪問 http://localhost:5001/api/debug/chunks_with_overlap 以查看文檔分塊結果的視覺化展示，可調整分塊大小和重疊度參數。

---

## 配置選項
主要配置選項位於 config.py：

- CHUNK_SIZE：文本分塊大小，影響檢索顆粒度
- CHUNK_OVERLAP：分塊重疊度，影響上下文連貫性
- TOP_K_RESULTS：檢索時返回的最相關文檔數量
- TEMPERATURE：生成模型的溫度參數
- MAX_TOKENS：生成回答的最大 token 限制