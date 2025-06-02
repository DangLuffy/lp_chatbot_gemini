# /app/api/handlers.py
"""
Định nghĩa Endpoint: Một endpoint POST /solve được tạo ra. Nó chấp nhận một yêu cầu có chứa problem_data (JSON) hoặc problem_text (văn bản).

Kiểm tra đầu vào:

Nếu problem_data tồn tại: 
    - Tuyệt vời! Dữ liệu đã có cấu trúc. API handler chỉ cần gán nó cho biến problem_dict để chuẩn bị giải.
Nếu problem_text tồn tại: 
    - Đây chính là lúc utils.py được sử dụng. Code sẽ gọi hàm parse_lp_problem_from_text(request.problem_text). 
    - Hàm này sẽ đọc chuỗi văn bản và (khi được triển khai đầy đủ) sẽ trả về một dictionary problem_dict có cấu trúc chuẩn.
Nếu không có cả hai: 
    - API sẽ báo lỗi cho người dùng.
Gọi bộ giải: 
    - Sau khi đã có problem_dict (dù là từ problem_data hay từ kết quả phân tích của utils.py), API handler sẽ gọi dispatch_solver để thực hiện việc giải bài toán.

Trả kết quả: Cuối cùng, API trả về kết quả từ bộ giải và toàn bộ logs (bao gồm cả log từ việc phân tích cú pháp, nếu có) cho người dùng.


"""
import logging
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List

# Import các thành phần cần thiết từ các module khác
from app.solver.dispatcher import dispatch_solver
from app.solver.utils import parse_lp_problem_from_text # <<<----- ĐÂY LÀ LÚC UTILS.PY ĐƯỢC SỬ DỤNG
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ---- Định nghĩa các mô hình dữ liệu cho Request Body ----
# Điều này giúp FastAPI tự động kiểm tra dữ liệu đầu vào

class SolveRequest(BaseModel):
    # Người dùng có thể cung cấp một trong hai:
    # 1. problem_data: một dictionary có cấu trúc sẵn
    # 2. problem_text: một chuỗi văn bản mô tả bài toán
    problem_data: Optional[Dict[str, Any]] = None
    problem_text: Optional[str] = None
    
    # Người dùng cũng có thể chọn bộ giải
    solver_name: str = "pulp_cbc" # Mặc định là pulp_cbc

class SolveResponse(BaseModel):
    solution: Optional[Dict[str, Any]]
    logs: List[str]
    message: str


# ---- Định nghĩa API Endpoint ----

@router.post("/solve", response_model=SolveResponse, summary="Solve a Linear Programming Problem")
async def solve_problem(request: SolveRequest):
    """
    Nhận một bài toán Quy hoạch tuyến tính và giải nó.

    Bạn có thể gửi bài toán theo một trong hai cách:
    - **problem_data**: Một đối tượng JSON có cấu trúc đầy đủ.
    - **problem_text**: Một chuỗi văn bản thô mô tả bài toán (parser hiện tại còn đơn giản).
    
    Bạn cũng có thể chỉ định `solver_name` ("pulp_cbc" hoặc "simplex_manual").
    """
    logger.info(f"Received request to solve problem with solver: {request.solver_name}")
    
    # Khởi tạo biến problem_dict để lưu trữ dữ liệu bài toán đã được chuẩn hóa
    problem_dict: Optional[Dict[str, Any]] = None
    logs = []

    # === BƯỚC QUAN TRỌNG: SỬ DỤNG UTILS.PY KHI CẦN THIẾT ===
    
    if request.problem_data:
        # Trường hợp 1: Người dùng gửi dữ liệu có cấu trúc sẵn
        logger.info("Using structured problem_data from request.")
        problem_dict = request.problem_data
        logs.append("Received structured 'problem_data'. No parsing needed.")

    elif request.problem_text:
        # Trường hợp 2: Người dùng gửi văn bản thô -> CẦN GỌI UTILS.PY
        logger.info("Received raw 'problem_text'. Attempting to parse...")
        logs.append("Parsing 'problem_text' into structured data...")
        
        # <<<----- GỌI HÀM TỪ UTILS.PY Ở ĐÂY ----->>>
        parsed_data, parse_logs = parse_lp_problem_from_text(request.problem_text)
        logs.extend(parse_logs)
        
        if not parsed_data:
            logger.error("Failed to parse problem_text.")
            # Lưu ý: Vì parser trong utils.py hiện đang là placeholder, nó sẽ luôn trả về lỗi ở đây.
            # Khi parser được triển khai đầy đủ, nó sẽ hoạt động.
            raise HTTPException(status_code=400, detail={"message": "Failed to parse 'problem_text'.", "logs": logs})
        
        problem_dict = parsed_data

    else:
        # Nếu không có dữ liệu nào được cung cấp
        logger.error("No problem_data or problem_text provided in the request.")
        raise HTTPException(status_code=400, detail="You must provide either 'problem_data' or 'problem_text'.")

    # === SAU KHI CÓ DỮ LIỆU CHUẨN, GỌI DISPATCHER ĐỂ GIẢI ===
    
    if problem_dict:
        logger.info(f"Dispatching problem to solver '{request.solver_name}'...")
        solution, solver_logs = dispatch_solver(problem_data=problem_dict, solver_name=request.solver_name)
        logs.extend(solver_logs)
        
        if solution:
            return SolveResponse(solution=solution, logs=logs, message=f"Problem solved successfully by {request.solver_name}.")
        else:
            # Nếu bộ giải không tìm thấy lời giải hoặc có lỗi
            return SolveResponse(solution=None, logs=logs, message=f"Solver {request.solver_name} failed to find a solution.")

    # Trường hợp không thể xảy ra nếu logic ở trên đúng, nhưng để an toàn
    raise HTTPException(status_code=500, detail="An unexpected error occurred.")

