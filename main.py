# /main.py
# Tệp khởi động chính của ứng dụng

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Import các thành phần từ các module đã tạo
from core.config import settings
from core.logger import setup_logging
from app.api.routes import api_router
from app.chatbot.web_routes import router as chatbot_web_router

# -- BƯỚC 1: THIẾT LẬP LOGGING --
# Phải được gọi ở đây, trước khi tạo instance của app,
# để đảm bảo logging được áp dụng cho toàn bộ ứng dụng ngay từ đầu.
setup_logging()
logger = logging.getLogger(__name__)


# -- BƯỚC 2: KHỞI TẠO ỨNG DỤNG FASTAPI --
logger.info("Initializing FastAPI application...")
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    debug=settings.DEBUG,
    # Cấu hình đường dẫn cho tài liệu API (Swagger UI)
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc"
)


# -- BƯỚC 3: GẮN (MOUNT) CÁC TỆP TĨNH --
# Dòng này rất quan trọng, nó cho phép FastAPI phục vụ các tệp CSS, JS, hình ảnh...
# từ thư mục 'static' tại đường dẫn '/static'.
# Ví dụ: trình duyệt có thể truy cập /static/css/style.css
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.info("Static files directory mounted at /static.")


# -- BƯỚC 4: BAO GỒM (INCLUDE) CÁC ROUTER --

# Bao gồm router cho các API RESTful (ví dụ: /api/v1/lp/solve)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["Backend API"])
logger.info(f"Included API router with prefix: {settings.API_V1_STR}")

# Bao gồm router cho giao diện web của chatbot
# Ở đây không đặt prefix để có thể truy cập trực tiếp qua /chat
# Điều này giúp đơn giản hóa URL trong file JavaScript (fetch('/send_message'))
app.include_router(chatbot_web_router, tags=["Chatbot Web UI"])
logger.info("Included Chatbot Web UI router at root.")

# -- BƯỚC 5: ĐỊNH NGHĨA ENDPOINT GỐC (ROOT) --
@app.get("/", include_in_schema=False)
async def read_root():
    """
    Endpoint gốc, tự động chuyển hướng người dùng đến giao diện chat.
    `include_in_schema=False` để ẩn nó khỏi tài liệu API.
    """
    logger.info("Root endpoint '/' accessed, redirecting to '/chat'.")
    return RedirectResponse(url="/chat")


# -- BƯỚC 6: KHỞI CHẠY SERVER (KHI CHẠY TRỰC TIẾP TỆP NÀY) --
if __name__ == "__main__":
    logger.info(f"Starting server on http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    uvicorn.run(
        "main:app",  # Tham chiếu đến đối tượng 'app' trong tệp 'main.py'
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level=settings.LOG_LEVEL.lower(), # Đặt log level cho uvicorn
        reload=settings.DEBUG  # reload=True rất hữu ích trong môi trường development
                               # Nó sẽ tự động khởi động lại server mỗi khi bạn lưu file code.
                               # Nên đặt là False trong môi trường production.
    )

