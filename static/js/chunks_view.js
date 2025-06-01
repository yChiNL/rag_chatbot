function reloadWithParams() {
    const chunkSize = document.getElementById('chunkSize').value;
    const overlapSize = document.getElementById('overlapSize').value;
    
    window.location.href = `/api/debug/chunks_with_overlap?chunk_size=${chunkSize}&overlap=${overlapSize}`;
}

function searchInChunks() {
    // 獲取搜索關鍵詞
    const searchText = document.getElementById('searchInput').value.trim().toLowerCase();
    
    // 獲取所有 chunk 內容元素
    const chunkContents = document.querySelectorAll('.chunk-content');
    
    // 如果搜索框為空，恢復所有原始內容
    if (searchText === '') {
        chunkContents.forEach(content => {
            // 重置為原始 HTML (保留重疊高亮)
            content.innerHTML = content.getAttribute('data-original-html') || content.innerHTML;
        });
        return;
    }
    
    // 遍歷每個 chunk 內容
    chunkContents.forEach(content => {
        // 如果尚未保存原始內容，則保存
        if (!content.getAttribute('data-original-html')) {
            content.setAttribute('data-original-html', content.innerHTML);
        }
        
        // 獲取原始 HTML
        let originalHtml = content.getAttribute('data-original-html');
        
        // 創建一個臨時 div 來處理 HTML
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = originalHtml;
        let plainText = tempDiv.textContent;
        
        // 檢查是否包含關鍵詞
        const lowerText = plainText.toLowerCase();
        if (lowerText.includes(searchText)) {
            // 高亮處理，同時保留原有的格式
            let lastIndex = 0;
            let result = '';
            let index;
            
            while ((index = lowerText.indexOf(searchText, lastIndex)) !== -1) {
                // 添加前面不匹配的部分
                result += originalHtml.substring(lastIndex, index);
                
                // 添加匹配的部分，使用高亮
                const matchedText = originalHtml.substr(index, searchText.length);
                result += `<span class="match-highlight">${matchedText}</span>`;
                
                lastIndex = index + searchText.length;
            }
            
            // 添加最後剩餘的部分
            if (lastIndex < originalHtml.length) {
                result += originalHtml.substring(lastIndex);
            }
            
            content.innerHTML = result;
            
            // 讓包含搜索結果的 chunk 滾動到視圖中
            content.parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            // 不包含關鍵詞，恢復原始內容
            content.innerHTML = originalHtml;
        }
    });
}

// 初始加載時保存所有原始 HTML
document.addEventListener('DOMContentLoaded', function() {
    const chunkContents = document.querySelectorAll('.chunk-content');
    chunkContents.forEach(content => {
        content.setAttribute('data-original-html', content.innerHTML);
    });
});