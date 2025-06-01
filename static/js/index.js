// DOM元素
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const searchResultsContainer = document.getElementById('searchResultsContainer');

// 查詢計數器
let queryCounter = 0;

// 發送消息
function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    // 計數器增加
    queryCounter++;
    
    // 添加用戶消息到聊天區
    addMessage(message, 'user');
    
    // 清空輸入框
    userInput.value = '';
    
    // 顯示加載中
    searchResultsContainer.innerHTML = '<div class="loading">正在檢索相關文檔...</div>';
    
    // 添加思考中的提示
    const thinkingMessage = document.createElement('div');
    thinkingMessage.className = 'system-message';
    thinkingMessage.innerText = '思考中...';
    chatMessages.appendChild(thinkingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // 禁用發送按鈕
    sendButton.disabled = true;
    
    // 發送到後端API
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message }),
    })
    .then(response => response.json())
    .then(data => {
        // 移除思考中提示
        if (chatMessages.contains(thinkingMessage)) {
            chatMessages.removeChild(thinkingMessage);
        }
        
        // 添加機器人回覆
        addMessage(data.answer, 'bot');
        
        // 顯示檢索結果
        displaySearchResults(data.docs, queryCounter);
    })
    .catch(error => {
        console.error('Error:', error);
        
        // 移除思考中提示
        if (chatMessages.contains(thinkingMessage)) {
            chatMessages.removeChild(thinkingMessage);
        }
        
        addMessage('抱歉，處理您的請求時發生錯誤。請稍後重試。', 'bot');
        searchResultsContainer.innerHTML = '<div class="no-results">檢索結果加載失敗</div>';
    })
    .finally(() => {
        // 啟用發送按鈕
        sendButton.disabled = false;
    });
}

// 添加消息到聊天區
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    // 創建消息內容元素
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // 處理機器人回答的 Markdown
    if (sender === 'bot') {
        // 使用 marked.js 解析 Markdown，並使用 DOMPurify 清理 HTML
        contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(text.trim()));
    } else {
        // 用戶消息不用 Markdown 解析
        contentDiv.textContent = text.trim();
    }
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = getCurrentTime();
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    chatMessages.appendChild(messageDiv);
    
    // 滾動到最新消息
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 顯示檢索結果
function displaySearchResults(docs, queryNum) {
    if (!docs || docs.length === 0) {
        searchResultsContainer.innerHTML = '<div class="no-results">未找到相關文檔</div>';
        return;
    }
    
    searchResultsContainer.innerHTML = `<div class="query-counter">查詢 #${queryNum} 的檢索結果</div>`;
    
    docs.forEach(doc => {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        const rankDiv = document.createElement('div');
        rankDiv.className = 'result-rank';
        rankDiv.textContent = `檢索結果 #${doc.rank}`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'result-text';
        textDiv.textContent = doc.text.trim();
        
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'result-source';
        sourceDiv.textContent = `來源: ${doc.source}`;
        
        resultItem.appendChild(rankDiv);
        resultItem.appendChild(textDiv);
        resultItem.appendChild(sourceDiv);
        
        searchResultsContainer.appendChild(resultItem);
    });
}

// 獲取當前時間
function getCurrentTime() {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
}

// 顯示響應式數據庫信息
function checkResponsive() {
    const dbInfoMobile = document.querySelector('.db-info-mobile');
    const dbInfo = document.querySelector('.db-info');
    
    if (window.innerWidth <= 768) {
        dbInfoMobile.style.display = 'flex';
        dbInfo.style.display = 'none';
    } else {
        dbInfoMobile.style.display = 'none';
        dbInfo.style.display = 'flex';
    }
}

// 事件監聽器
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// 檢查響應式布局
window.addEventListener('load', checkResponsive);
window.addEventListener('resize', checkResponsive);

// 初始聚焦到輸入框
userInput.focus();