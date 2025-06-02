# /app/chatbot/web_routes.py

import logging
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from .dialog_manager import DialogManager # Import DialogManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Thiết lập đường dẫn đến thư mục templates
# Giả sử thư mục gốc của dự án là nơi chứa thư mục 'app'
# BASE_DIR = Path(__file__).resolve().parent.parent.parent # Lên 3 cấp: web_routes.py -> chatbot -> app -> project_root
# TEMPLATES_DIR = BASE_DIR / "app" / "chatbot" / "templates"

# Cách đơn giản hơn nếu cấu trúc được biết rõ:
# Giả sử tệp này nằm trong app/chatbot, templates nằm trong app/chatbot/templates
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

if not TEMPLATES_DIR.exists():
    logger.error(f"Templates directory not found at: {TEMPLATES_DIR}")
    # Trong trường hợp thực tế, bạn có thể muốn raise lỗi hoặc có cơ chế fallback
    # For now, we'll proceed, but Jinja2Templates will fail if dir doesn't exist.

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Dependency để lấy DialogManager (có thể mở rộng để quản lý user session sau này)
def get_dialog_manager():
    # Trong ví dụ này, mỗi request sẽ tạo một instance mới hoặc dùng chung
    # Để có session người dùng riêng, bạn cần logic phức tạp hơn (ví dụ: dựa trên cookie, token)
    return DialogManager(user_id="web_chat_user")

@router.get("/chat", response_class=HTMLResponse, summary="Giao diện chat với LP Chatbot")
async def get_chat_interface(request: Request):
    """
    Hiển thị giao diện chat HTML.
    """
    logger.info(f"Serving chat interface from template directory: {TEMPLATES_DIR}")
    # Kiểm tra xem template index.html có tồn tại không
    index_template_path = TEMPLATES_DIR / "index.html"
    if not index_template_path.is_file():
        logger.error(f"index.html not found in {TEMPLATES_DIR}")
        return HTMLResponse(content="<h1>Lỗi: Không tìm thấy tệp index.html</h1>", status_code=500)
        
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/send_message", summary="Gửi tin nhắn đến chatbot và nhận phản hồi")
async def send_message_to_bot(
    message: str = Form(...), 
    dm: DialogManager = Depends(get_dialog_manager)
):
    """
    Nhận tin nhắn từ người dùng qua form, xử lý bằng DialogManager và trả về phản hồi.
    """
    logger.info(f"Received message via POST: '{message}'")
    if not message.strip():
        return {"bot_response": "Vui lòng nhập gì đó!", "logs": []}
        
    bot_response_text = dm.handle_message(message)
    # Lấy logs từ dialog manager nếu bạn muốn hiển thị chúng (ví dụ cho mục đích debug)
    # current_logs = dm.get_logs() 
    
    return {"bot_response": bot_response_text}

# Để sử dụng router này, bạn cần include nó vào app FastAPI chính trong main.py:
# from app.chatbot.web_routes import router as chatbot_web_router
# app.include_router(chatbot_web_router, tags=["Chatbot Interface"])
