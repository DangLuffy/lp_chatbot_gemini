# /app/chatbot/web_routes.py

import logging
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse # Thêm JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from .dialog_manager import DialogManager 

logger = logging.getLogger(__name__)
router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

if not TEMPLATES_DIR.exists():
    logger.error(f"Templates directory not found at: {TEMPLATES_DIR}")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Tạo một instance DialogManager duy nhất để chia sẻ (hoặc dùng DI phức tạp hơn)
# Lưu ý: Cách này không thread-safe cho nhiều user cùng lúc nếu DialogManager lưu trạng thái user.
# Cần cơ chế quản lý session người dùng riêng biệt cho ứng dụng thực tế.
# Hiện tại, user_id đang được hardcode là "web_chat_user".
dialog_manager_instance = DialogManager(user_id="web_chat_user_main_instance")

def get_dialog_manager():
    # return DialogManager(user_id="web_chat_user") # Tạo mới mỗi request
    return dialog_manager_instance # Dùng chung instance (cẩn thận với state)

@router.get("/chat", response_class=HTMLResponse, summary="Giao diện chat với LP Chatbot")
async def get_chat_interface(request: Request):
    logger.info(f"Serving chat interface from template directory: {TEMPLATES_DIR}")
    index_template_path = TEMPLATES_DIR / "index.html"
    if not index_template_path.is_file():
        logger.error(f"index.html not found in {TEMPLATES_DIR}")
        return HTMLResponse(content="<h1>Lỗi: Không tìm thấy tệp index.html</h1>", status_code=500)
        
    return templates.TemplateResponse("index.html", {"request": request})

# !!! QUAN TRỌNG: Đổi send_message_to_bot thành async def !!!
@router.post("/send_message", summary="Gửi tin nhắn đến chatbot và nhận phản hồi", response_class=JSONResponse)
async def send_message_to_bot(
    message: str = Form(...), 
    dm: DialogManager = Depends(get_dialog_manager) # Sử dụng Depends để lấy instance
):
    logger.info(f"Received message via POST: '{message}' for user '{dm.user_id}'") # Log user_id
    if not message.strip():
        return {"bot_response": {"text_response": "Vui lòng nhập gì đó!", "plot_image_base64": None, "suggestions": []}}
        
    # Gọi hàm handle_message bất đồng bộ
    bot_response_data = await dm.handle_message(message) 
    
    # Trả về toàn bộ dictionary bot_response_data
    # Frontend (JavaScript) sẽ xử lý object này
    return {"bot_response": bot_response_data}

@router.post("/reset_chat_session", summary="Reset trạng thái hội thoại của chatbot cho user hiện tại")
async def reset_session(dm: DialogManager = Depends(get_dialog_manager)):
    logger.info(f"Resetting chat session for user '{dm.user_id}'.")
    dm.reset_state()
    initial_message = dm.state.get("last_bot_message", "Đã làm mới. Bạn muốn bắt đầu lại chứ?")
    return {"bot_response": {"text_response": initial_message, "plot_image_base64": None, "suggestions": ["Nhập bài toán mới"]}}

