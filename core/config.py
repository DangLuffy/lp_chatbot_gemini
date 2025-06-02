# /core/config.py

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List

# Tải các biến môi trường từ tệp .env (nếu có)
# Điều này hữu ích cho việc quản lý các cấu hình nhạy cảm trong môi trường development.
# Trong production, các biến môi trường thường được thiết lập trực tiếp trên server.
load_dotenv()

class Settings(BaseSettings):
    """
    Lớp quản lý cấu hình chung cho ứng dụng.
    Các giá trị sẽ được đọc từ biến môi trường hoặc giá trị mặc định.
    """
    PROJECT_NAME: str = "Linear Programming Chatbot"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Cấu hình Server (ví dụ cho Uvicorn)
    SERVER_HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

    # Cấu hình API
    API_V1_STR: str = "/api/v1"

    # Cấu hình Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Ví dụ về các cấu hình khác có thể cần
    # DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    # NLP_MODEL_PATH: str = os.getenv("NLP_MODEL_PATH", "models/nlp_model.pkl")
    # ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"] # Cho CORS

    class Config:
        # Tên tệp .env để tải biến môi trường (nếu bạn muốn đổi tên)
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Cho phép Pydantic phân biệt chữ hoa/thường cho tên biến môi trường
        case_sensitive = True

# Tạo một instance của Settings để sử dụng trong toàn bộ ứng dụng
settings = Settings()

# In ra một vài cấu hình khi module được tải (chỉ cho mục đích debug)
if __name__ == "__main__":
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Server Host: {settings.SERVER_HOST}")
    print(f"Server Port: {settings.SERVER_PORT}")
    print(f"Log Level: {settings.LOG_LEVEL}")
