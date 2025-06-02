# /app/api/routes.py
"""
Tệp app/api/routes.py này tạo một APIRouter chính tên là api_router. Sau đó, 
nó include (bao gồm) solve_router (được import từ app.api.handlers.py mà bạn đã chọn) 
vào api_router này với một tiền tố (prefix) là /lp và một tag là "Linear Programming Solver".

Điều này có nghĩa là endpoint /solve được định nghĩa trong handlers.py sẽ có đường dẫn đầy đủ là 
/api/v1/lp/solve (giả sử api_router được include vào app chính với prefix /api/v1 như trong main.py).

Bạn có thể thêm các router từ các tệp handlers khác vào api_router này nếu ứng dụng 
của bạn có nhiều tính năng hơn.

"""
from fastapi import APIRouter

# Import router từ các tệp handlers
# Ví dụ, nếu bạn có một tệp handlers.py chứa logic cho endpoint "/solve"
from .handlers import router as solve_router

# Bạn có thể có các tệp handlers khác cho các nhóm endpoint khác nhau
# from .another_handlers import router as another_router

# Khởi tạo một APIRouter chính cho toàn bộ API version 1 (hoặc một nhóm logic cụ thể)
api_router = APIRouter()

# Bao gồm các router con vào router chính
# Mỗi router con có thể có một prefix riêng nếu cần
api_router.include_router(solve_router, prefix="/lp", tags=["Linear Programming Solver"])
# Ví dụ:
# api_router.include_router(another_router, prefix="/other_feature", tags=["Other Feature"])

# Bạn có thể thêm các endpoint trực tiếp vào api_router ở đây nếu chúng đơn giản
# @api_router.get("/health", tags=["General"])
# async def health_check():
#     return {"status": "ok"}

# Tệp main.py sẽ import 'api_router' này và include nó vào FastAPI app chính.
