# /app/solver/utils.py
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging # Thêm import logging

logger = logging.getLogger(__name__) # Thêm logger cho utils

def normalize_problem_data_from_nlp(problem_data_input: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Chuẩn hóa problem_data từ định dạng NLP (mới) sang định dạng mà các solver mong đợi (cũ).
    Định dạng mới (ví dụ từ NLP):
    {
        "objective": "max",
        "coeffs": [c1, c2], # Hệ số hàm mục tiêu
        "variables_names_for_title_only": ["var1", "var2"], # Tùy chọn, để tạo tiêu đề đẹp
        "constraints": [ {"name": "R1", "lhs": [a1, a2], "op": "<=", "rhs": b1}, ... ]
    }
    Định dạng solver mong đợi:
    {
        "objective": {"type": "maximize", "coefficients": [c1, c2]},
        "variables": ["var1", "var2"], # Hoặc ["x1", "x2"] nếu không có tên tường minh
        "constraints": [ {"name": "R1", "coefficients": [a1, a2], "type": "<=", "rhs": b1}, ... ]
    }
    """
    logs = []
    # Kiểm tra xem có phải định dạng mới không (ví dụ: "coeffs" nằm ở cấp gốc và "objective" là string)
    if "coeffs" in problem_data_input and isinstance(problem_data_input.get("objective"), str):
        logs.append("Utils: Normalizing input data from new NLP format.")
        normalized_data: Dict[str, Any] = {"objective": {}, "variables": [], "constraints": []}
        
        # 1. Xử lý Objective
        obj_type_str = problem_data_input["objective"].lower()
        if obj_type_str == "max":
            normalized_data["objective"]["type"] = "maximize"
        elif obj_type_str == "min":
            normalized_data["objective"]["type"] = "minimize"
        else:
            logs.append(f"ERROR (Utils): Invalid objective type '{obj_type_str}' in new format.")
            return None, logs
            
        obj_coeffs = problem_data_input.get("coeffs", [])
        if not isinstance(obj_coeffs, list) or not all(isinstance(c, (int, float)) for c in obj_coeffs):
            logs.append("ERROR (Utils): 'coeffs' for objective must be a list of numbers in new format.")
            return None, logs
        normalized_data["objective"]["coefficients"] = obj_coeffs
        
        # 2. Xác định số lượng biến và tạo tên biến
        num_vars = len(obj_coeffs)
        
        # Cố gắng suy ra số biến từ ràng buộc đầu tiên nếu obj_coeffs rỗng
        # và problem_data_input.get("constraints") là một list và không rỗng
        if num_vars == 0 and \
           isinstance(problem_data_input.get("constraints"), list) and \
           problem_data_input["constraints"]:
            first_constr = problem_data_input["constraints"][0]
            if isinstance(first_constr, dict) and "lhs" in first_constr and isinstance(first_constr["lhs"], list):
                num_vars = len(first_constr["lhs"])
        
        if num_vars == 0: 
            # Nếu không có ràng buộc nào, và cũng không có hệ số mục tiêu, 
            # có thể là một bài toán không có biến (ví dụ: max z = 5, không ràng buộc)
            # Tuy nhiên, các solver hiện tại đều cần ít nhất một biến.
            # Hoặc nếu NLP trả về một "variables_names_for_title_only" rỗng
            if "variables_names_for_title_only" in problem_data_input and not problem_data_input["variables_names_for_title_only"]:
                logs.append("Warning (Utils): No variables defined by objective coeffs or constraints, and 'variables_names_for_title_only' is empty.")
                # Vẫn cho phép đi tiếp nếu người dùng muốn định nghĩa problem không biến (rất hiếm)
            elif not ("variables_names_for_title_only" in problem_data_input):
                 logs.append("ERROR (Utils): Cannot determine number of variables from new format.")
                 return None, logs


        # Sử dụng tên biến từ "variables_names_for_title_only" nếu có, nếu không thì tự tạo x1, x2...
        # Đảm bảo danh sách này có đủ tên cho num_vars
        provided_var_names = problem_data_input.get("variables_names_for_title_only", [])
        if not isinstance(provided_var_names, list): provided_var_names = []

        final_var_names = [f"x{i+1}" for i in range(num_vars)]
        for i in range(min(num_vars, len(provided_var_names))):
            if isinstance(provided_var_names[i], str):
                 final_var_names[i] = provided_var_names[i]
            else: # Nếu tên biến không phải string, dùng tên mặc định
                 logs.append(f"Warning (Utils): Invalid variable name at index {i} in 'variables_names_for_title_only', using default 'x{i+1}'.")

        normalized_data["variables"] = final_var_names
        
        # 3. Xử lý Constraints
        raw_constraints = problem_data_input.get("constraints", [])
        if not isinstance(raw_constraints, list):
            logs.append("ERROR (Utils): 'constraints' must be a list in new format.")
            return None, logs

        for i, constr_new in enumerate(raw_constraints):
            if not isinstance(constr_new, dict):
                logs.append(f"ERROR (Utils): Constraint at index {i} is not a dictionary.")
                return None, logs

            lhs = constr_new.get("lhs")
            op = constr_new.get("op")
            rhs = constr_new.get("rhs")

            if not isinstance(lhs, list) or not all(isinstance(c, (int, float)) for c in lhs):
                logs.append(f"ERROR (Utils): 'lhs' in constraint {i+1} must be a list of numbers.")
                return None, logs
            if not isinstance(op, str):
                logs.append(f"ERROR (Utils): 'op' in constraint {i+1} must be a string.")
                return None, logs
            if not isinstance(rhs, (int, float)):
                logs.append(f"ERROR (Utils): 'rhs' in constraint {i+1} must be a number.")
                return None, logs
            
            if num_vars > 0 and len(lhs) != num_vars: # Chỉ kiểm tra nếu num_vars > 0
                logs.append(f"ERROR (Utils): Number of coefficients in LHS of constraint {i+1} ({len(lhs)}) does not match number of variables ({num_vars}).")
                return None, logs

            normalized_data["constraints"].append({
                "name": constr_new.get("name", f"c{i+1}"), 
                "coefficients": lhs,
                "type": op,
                "rhs": rhs
            })
        logs.append("Utils: Normalization successful.")
        return normalized_data, logs
    
    logs.append("Utils: Input data appears to be in the expected (old) format or is not a known new format.")
    # Thêm kiểm tra cơ bản cho định dạng cũ để đảm bảo tính nhất quán
    if not (isinstance(problem_data_input.get("objective"), dict) and \
            "type" in problem_data_input["objective"] and \
            "coefficients" in problem_data_input["objective"] and \
            isinstance(problem_data_input.get("variables"), list) and \
            isinstance(problem_data_input.get("constraints"), list)):
        logs.append("ERROR (Utils): Data does not match expected old format either.")
        return None, logs # Trả về None nếu định dạng cũ cũng không hợp lệ
        
    return problem_data_input, logs


def parse_lp_problem_from_text(text_input: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Phân tích cú pháp một bài toán LP từ dạng văn bản đơn giản.
    Đây là một trình phân tích cú pháp rất cơ bản và có thể cần mở rộng đáng kể
    """
    # ... (Nội dung hàm này giữ nguyên như trước) ...
    logs = ["Starting to parse LP problem from text."]
    problem_data: Dict[str, Any] = {
        "objective": {"type": "", "coefficients": []},
        "variables": [],
        "constraints": [],
        "bounds": {}
    }
    lines = [line.strip() for line in text_input.split('\n') if line.strip()]
    if not lines:
        logs.append("Error: Input text is empty.")
        return None, logs
    logs.append("Warning: Text parsing (parse_lp_problem_from_text) is currently a placeholder and not fully implemented.")
    logs.append("Please provide problem_data directly in dictionary format for now, or use the new NLP format if testing normalization.")
    return None, logs # Trả về lỗi vì chưa triển khai đầy đủ


def convert_problem_to_matrix_form(problem_data: Dict[str, Any]) -> Optional[Dict[str, Union[np.ndarray, str, List[str]]]]:
    """
    Chuyển đổi problem_data sang dạng ma trận (A, b, c).
    """
    # ... (Nội dung hàm này giữ nguyên như trước) ...
    try:
        # Đảm bảo problem_data ở định dạng "cũ" chuẩn trước khi chuyển đổi
        # Nếu không, bạn có thể gọi hàm chuẩn hóa ở đây, nhưng thường thì hàm này
        # được kỳ vọng nhận đầu vào đã chuẩn.
        if not (isinstance(problem_data.get("objective"), dict) and \
            "coefficients" in problem_data["objective"] and \
            isinstance(problem_data.get("variables"), list)):
             logger.warning("convert_problem_to_matrix_form: input problem_data might not be in standard solver format.")
             # Có thể raise lỗi hoặc cố gắng chuyển đổi, tùy theo yêu cầu.
             # Hiện tại, sẽ cố gắng tiếp tục.

        c_vector = np.array(problem_data["objective"]["coefficients"], dtype=float)
        variable_names = list(problem_data["variables"]) 
        num_vars = len(variable_names)

        constraints = problem_data["constraints"]
        num_constraints = len(constraints)

        A_matrix = np.zeros((num_constraints, num_vars), dtype=float)
        b_vector = np.zeros(num_constraints, dtype=float)
        constraint_types = []

        for i, constr in enumerate(constraints):
            if len(constr["coefficients"]) != num_vars:
                raise ValueError(f"Constraint '{constr.get('name', i)}' has incorrect number of coefficients.")
            A_matrix[i, :] = constr["coefficients"]
            b_vector[i] = constr["rhs"]
            constraint_types.append(constr["type"])
        
        return {
            "c": c_vector,
            "A": A_matrix,
            "b": b_vector,
            "constraint_types": constraint_types,
            "objective_type": problem_data["objective"]["type"],
            "variable_names": variable_names
        }
    except Exception as e:
        logger.error(f"Error converting problem to matrix form: {e}")
        return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Để DEBUG để xem log từ normalize

    # Test hàm normalize_problem_data_from_nlp
    new_format_test = {
        "objective": "max",
        "coeffs": [3, 2, 5],
        "variables_names_for_title_only": ["apple", "banana", "cherry"],
        "constraints": [
            {"lhs": [1, 1, 0], "op": "<=", "rhs": 10, "name": "ConstraintAlpha"},
            {"lhs": [0, 2, 1], "op": "≥", "rhs": 15} # op sẽ được giữ nguyên
        ]
    }
    print("--- Testing Normalization ---")
    normalized, norm_logs = normalize_problem_data_from_nlp(new_format_test)
    for log in norm_logs:
        print(log)
    if normalized:
        import json
        print("Normalized Data:\n", json.dumps(normalized, indent=2))

    print("\n--- Testing Normalization with empty objective coeffs (should take from constraints) ---")
    new_format_empty_obj = {
        "objective": "min",
        "coeffs": [],
        "constraints": [
            {"lhs": [1, -1], "op": "==", "rhs": 0}
        ]
    }
    normalized_empty, norm_logs_empty = normalize_problem_data_from_nlp(new_format_empty_obj)
    for log in norm_logs_empty: print(log)
    if normalized_empty: print("Normalized Empty Obj Data:\n", json.dumps(normalized_empty, indent=2))


    # Test hàm convert_problem_to_matrix_form với dữ liệu đã chuẩn hóa
    if normalized: # Nếu việc chuẩn hóa ở trên thành công
        print("\n--- Converting Normalized Dict to Matrix Form ---")
        matrix_data = convert_problem_to_matrix_form(normalized)
        if matrix_data:
            for k, v_item in matrix_data.items(): # Đổi tên biến lặp
                print(f"{k}:")
                if isinstance(v_item, np.ndarray):
                    print(v_item)
                else:
                    print(v_item)
        else:
            print("Failed to convert to matrix form.")
