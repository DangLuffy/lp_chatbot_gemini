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

# Import các thành phần cần thiết
from app.solver.dispatcher import dispatch_solver
from app.chatbot.nlp.lp_parser import parse_lp_problem_from_string # Parser mới
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ---- Định nghĩa các mô hình dữ liệu cho Request Body ----
class SolveRequest(BaseModel):
    problem_data: Optional[Dict[str, Any]] = None # Mong đợi "Định dạng A" nếu được cung cấp
    problem_text: Optional[str] = None
    solver_name: str = "pulp_cbc"
    max_iterations: int = 50 # Thêm max_iterations

class SolveResponse(BaseModel):
    solution: Optional[Dict[str, Any]]
    logs: List[str]
    message: str

# --- Hàm tiện ích để chuyển đổi từ định dạng coeffs_map sang "Định dạng A" ---
def _convert_coeffs_map_to_format_a(
    problem_data_coeffs_map: Dict[str, Any],
    logs: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Chuyển đổi cấu trúc dữ liệu bài toán từ dạng sử dụng coeffs_map (output của lp_parser)
    sang "Định dạng A" (sử dụng list cho coeffs và lhs).
    """
    if not problem_data_coeffs_map or not problem_data_coeffs_map.get("objective_type"):
        logs.append("ERROR (_convert_coeffs_map_to_format_a): Dữ liệu đầu vào không đủ (thiếu objective_type).")
        return None

    objective_type = problem_data_coeffs_map.get("objective_type")
    objective_coeffs_map_internal = problem_data_coeffs_map.get("objective_coeffs_map", {})
    constraints_internal = problem_data_coeffs_map.get("constraints", []) # Đây là list các ràng buộc, mỗi cái có coeffs_map

    # Xác định một thứ tự nhất quán cho các biến
    all_vars_set = set(objective_coeffs_map_internal.keys())
    for constr_map_item in constraints_internal: # Đổi tên biến lặp
        all_vars_set.update(constr_map_item.get("coeffs_map", {}).keys())
    
    final_variables_ordered = sorted(list(all_vars_set))

    if not final_variables_ordered and objective_coeffs_map_internal:
        # Trường hợp chỉ có hàm mục tiêu với hằng số, không có biến (hiếm)
        logs.append("WARNING (_convert_coeffs_map_to_format_a): Không tìm thấy biến nào nhưng objective_coeffs_map tồn tại.")
    elif not final_variables_ordered and not objective_coeffs_map_internal:
        # Nếu không có biến nào và không có hệ số mục tiêu, có thể là bài toán không hợp lệ nếu có ràng buộc.
        # Tuy nhiên, nếu không có ràng buộc thì có thể là bài toán chỉ có hằng số.
        # Đối với mục đích API, nếu không có biến nào được tìm thấy, coi như lỗi trừ khi đó là bài toán rỗng.
        if constraints_internal: # Nếu có ràng buộc mà không có biến thì là lỗi
            logs.append("ERROR (_convert_coeffs_map_to_format_a): Không tìm thấy biến nào trong mục tiêu hoặc ràng buộc.")
            return None


    # Chuyển đổi hàm mục tiêu
    objective_coeffs_list_format_a = [objective_coeffs_map_internal.get(var, 0.0) for var in final_variables_ordered] # Đổi tên biến

    # Chuyển đổi các ràng buộc
    format_a_constraints_list = [] # Đổi tên biến
    for i, constr_map_item_convert in enumerate(constraints_internal): # Đổi tên biến lặp
        lhs_coeffs_list_format_a = [constr_map_item_convert.get("coeffs_map", {}).get(var, 0.0) for var in final_variables_ordered] # Đổi tên biến
        
        op_val = constr_map_item_convert.get("operator") # Đổi tên biến
        rhs_val = constr_map_item_convert.get("rhs")     # Đổi tên biến

        if op_val is None or rhs_val is None:
            logs.append(f"ERROR (_convert_coeffs_map_to_format_a): Ràng buộc '{constr_map_item_convert.get('name', i)}' thiếu 'operator' hoặc 'rhs'.")
            return None

        format_a_constraints_list.append({
            "name": constr_map_item_convert.get("name", f"api_c{i+1}"),
            "lhs": lhs_coeffs_list_format_a,
            "op": op_val,
            "rhs": rhs_val
        })

    format_a_problem_dict = { # Đổi tên biến
        "objective": objective_type,
        "coeffs": objective_coeffs_list_format_a,
        "variables_names_for_title_only": final_variables_ordered,
        "constraints": format_a_constraints_list
        # "bounds" không được xử lý bởi lp_parser hiện tại, nên sẽ không có ở đây.
    }
    logs.append(f"Đã chuyển đổi thành công cấu trúc coeffs_map sang Định dạng A. Các biến: {final_variables_ordered}")
    return format_a_problem_dict

# ---- Định nghĩa API Endpoint ----

@router.post("/solve", response_model=SolveResponse, summary="Giải một bài toán Quy hoạch Tuyến tính")
async def solve_problem_api(request: SolveRequest = Body(...)): # Đổi tên hàm để tránh trùng, thêm Body
    """
    Nhận một bài toán Quy hoạch tuyến tính và giải nó.

    Bạn có thể gửi bài toán theo một trong hai cách:
    - **problem_data**: Một đối tượng JSON có cấu trúc "Định dạng A".
    - **problem_text**: Một chuỗi văn bản thô mô tả bài toán.
    
    Bạn cũng có thể chỉ định `solver_name` và `max_iterations`.
    """
    request_data = request.model_dump() # Sử dụng model_dump cho Pydantic v2+
    solver_name_req = request_data.get("solver_name", "pulp_cbc") # Đổi tên biến
    max_iterations_req = request_data.get("max_iterations", 50) # Đổi tên biến
    logger.info(f"Received API request to solve problem with solver: {solver_name_req}, max_iterations: {max_iterations_req}")
    
    problem_dict_format_a: Optional[Dict[str, Any]] = None # Đổi tên biến
    logs: List[str] = []

    if request_data.get("problem_data"):
        logger.info("Using structured problem_data from API request (expected as Format A).")
        problem_dict_format_a = request_data["problem_data"]
        logs.append("Received structured 'problem_data'. Assuming it is in 'Format A'.")
        # Thêm kiểm tra sơ bộ xem problem_data có vẻ giống "Định dạng A" không
        if not all(k in problem_dict_format_a for k in ["objective", "coeffs", "variables_names_for_title_only", "constraints"]):
            logs.append("Warning: 'problem_data' might not be in the expected 'Format A'. Missing some keys.")
            # Có thể raise HTTPException ở đây nếu muốn chặt chẽ
            # raise HTTPException(status_code=400, detail={"message": "Provided 'problem_data' does not seem to be in 'Format A'.", "logs": logs})


    elif request_data.get("problem_text"):
        logger.info("Received raw 'problem_text'. Attempting to parse...")
        logs.append("Parsing 'problem_text' into structured data (coeffs_map format)...")
        
        parsed_data_coeffs_map, parse_logs = parse_lp_problem_from_string(request_data["problem_text"])
        logs.extend(parse_logs)
        
        if not parsed_data_coeffs_map:
            logger.error("Failed to parse problem_text using lp_parser.")
            raise HTTPException(status_code=400, detail={"message": "Failed to parse 'problem_text'.", "logs": logs})
        
        logs.append("Converting parsed data (coeffs_map) to 'Format A'...")
        problem_dict_format_a = _convert_coeffs_map_to_format_a(parsed_data_coeffs_map, logs)
        
        if not problem_dict_format_a:
            logger.error("Failed to convert parsed data to 'Format A'.")
            raise HTTPException(status_code=400, detail={"message": "Failed to convert parsed text data to required solver format.", "logs": logs})

    else:
        logger.error("No problem_data or problem_text provided in the API request.")
        raise HTTPException(status_code=400, detail="You must provide either 'problem_data' or 'problem_text'.")

    if problem_dict_format_a:
        logger.info(f"Dispatching problem (Format A) to solver '{solver_name_req}'...")
        solution, solver_logs = dispatch_solver(
            problem_data=problem_dict_format_a,
            solver_name=solver_name_req,
            max_iterations=max_iterations_req
        )
        logs.extend(solver_logs)
        
        if solution:
            return SolveResponse(solution=solution, logs=logs, message=f"Problem solved successfully by {solver_name_req}.")
        else:
            return SolveResponse(solution=None, logs=logs, message=f"Solver {solver_name_req} failed to find a solution or problem is infeasible/unbounded.")

    # Trường hợp không thể xảy ra nếu logic ở trên đúng
    logger.error("Unexpected state in API handler: problem_dict_format_a is None but no exception was raised.")
    raise HTTPException(status_code=500, detail="An unexpected error occurred in API handler.")

