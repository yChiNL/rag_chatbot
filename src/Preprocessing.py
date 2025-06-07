# Preprocessing.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataMiningProcessor:
    @staticmethod
    def preprocess_csv(file_path):
        """
        處理CSV檔案的資料前處理
        
        步驟：
        1. 刪除Brand欄位為空的行
        2. 其他欄位空值補"unknown"
        3. Weight Capacity (kg)和Price四捨五入到小數點第二位
        4. 使用metadata方式將標頭存於每筆value
        """
        try:
            # 讀取CSV
            logger.info(f"開始處理CSV文件: {file_path}")
            df = pd.read_csv(file_path)
            
            # 1. 刪除Brand欄位為空的行
            original_count = len(df)
            df = df.dropna(subset=['Brand'])
            logger.info(f"刪除Brand欄位為空的行: {original_count - len(df)}行被刪除")
            
            # 2. 其他欄位空值補"unknown"
            df = df.fillna("unknown")
            
            # 3. Weight Capacity (kg)和Price四捨五入到小數點第二位
            if 'Weight Capacity (kg)' in df.columns:
                df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].apply(
                    lambda x: round(float(x), 2) if x != "unknown" else x
                )
                
            if 'Price' in df.columns:
                df['Price'] = df['Price'].apply(
                    lambda x: round(float(x), 2) if x != "unknown" else x
                )
            
            # 4. 準備好的資料就直接轉為有metadata的文字格式
            processed_text = ""
            header_list = df.columns.tolist()
            
            for _, row in df.iterrows():
                row_data = []
                for col in header_list:
                    row_data.append(f"{col}: {row[col]}")
                processed_text += " | ".join(row_data) + "\n"
            
            logger.info(f"CSV文件 {file_path} 處理完成")
            return processed_text
            
        except Exception as e:
            logger.error(f"處理CSV文件時發生錯誤: {str(e)}", exc_info=True)
            return ""