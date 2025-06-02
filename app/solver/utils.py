# /app/solver/utils.py
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

def parse_lp_problem_from_text(text_input: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Phân tích cú pháp một bài toán LP từ dạng văn bản đơn giản.
    Đây là một trình phân tích cú pháp rất cơ bản và có thể cần mở rộng đáng kể
    để xử lý các định dạng phức tạp hơn hoặc sử dụng thư viện NLP.

    Định dạng văn bản mẫu dự kiến:
    Maximize (hoặc Minimize)
    obj: 3x1 + 2x2
    Subject to:
    c1: x1 + x2 <= 4
    c2: 2x1 + x2 <= 5
    Bounds: (Tùy chọn)
    x1 >= 0
    x2 >= 0
    x1 <= 10

    Returns:
        Tuple (problem_data, logs)
        problem_data: Dictionary cấu trúc giống như input của các solver.
        logs: Danh sách các thông báo log.
    """
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

    # Phần này cần một trình phân tích cú pháp mạnh mẽ hơn.
    # Ví dụ này chỉ là một cách tiếp cận đơn giản hóa.
    # Bạn có thể cân nhắc sử_dụng regex hoặc thư viện parsing chuyên dụng.
    # Hoặc tích hợp với module NLP.

    # TODO: Implement a robust parser.
    # For now, this function is a placeholder.
    logs.append("Warning: Text parsing is currently a placeholder and not fully implemented.")
    logs.append("Please provide problem_data directly in dictionary format for now.")
    
    # Giả sử một cấu trúc rất đơn giản để minh họa
    # Ví dụ:
    # Maximize
    # 3x1 + 2x2
    # x1 + x2 <= 4
    # 2x1 + x2 <= 5
    # x1 >= 0
    # x2 >= 0

    # Đây là một ví dụ rất thô sơ, không nên dùng trong thực tế mà không cải tiến.
    try:
        # Xác định loại hàm mục tiêu
        if lines[0].lower().startswith("maximize"):
            problem_data["objective"]["type"] = "maximize"
        elif lines[0].lower().startswith("minimize"):
            problem_data["objective"]["type"] = "minimize"
        else:
            logs.append("Error: Could not determine objective type (Maximize/Minimize) from the first line.")
            return None, logs
        logs.append(f"Objective type: {problem_data['objective']['type']}")

        # Phân tích hàm mục tiêu (giả sử dòng thứ 2)
        # Cần xác định biến và hệ số từ chuỗi như "3x1 + 2x2"
        # Đây là phần phức tạp, cần regex hoặc logic phân tích chuỗi tinh vi.
        # Ví dụ: objective_str = lines[1]
        # ... logic to parse objective_str ...
        # problem_data["variables"] = ["x1", "x2"] # Tạm thời hardcode
        # problem_data["objective"]["coefficients"] = [3, 2] # Tạm thời hardcode

        # Phân tích ràng buộc (giả sử các dòng tiếp theo)
        # ... logic to parse constraints ...
        # Ví dụ:
        # constraint1_str = lines[2] -> "x1 + x2 <= 4"
        # problem_data["constraints"].append({
        #     "name": "c1", "coefficients": [1,1], "type": "<=", "rhs": 4
        # })

        logs.append("Text parsing is highly simplified. Consider using a structured input or a dedicated NLP module.")
        # Trả về lỗi vì chưa triển khai đầy đủ
        return None, logs

    except IndexError:
        logs.append("Error: Input text format is not as expected or is incomplete.")
        return None, logs
    except Exception as e:
        logs.append(f"Error during text parsing: {e}")
        return None, logs

    # return problem_data, logs # Sẽ trả về khi triển khai xong

def convert_problem_to_matrix_form(problem_data: Dict[str, Any]) -> Optional[Dict[str, Union[np.ndarray, str, List[str]]]]:
    """
    Chuyển đổi problem_data sang dạng ma trận (A, b, c).
    Hữu ích cho một số thuật toán hoặc để hiển thị.

    Returns:
        Một dictionary chứa:
        'c': vector hệ số hàm mục tiêu (1D array)
        'A': ma trận hệ số ràng buộc (2D array)
        'b': vector vế phải ràng buộc (1D array)
        'constraint_types': danh sách các loại ràng buộc (e.g., ['<=', '>=', '=='])
        'objective_type': 'maximize' or 'minimize'
        'variable_names': danh sách tên biến
    """
    try:
        c = np.array(problem_data["objective"]["coefficients"], dtype=float)
        variable_names = list(problem_data["variables"]) # Đảm bảo là list
        num_vars = len(variable_names)

        constraints = problem_data["constraints"]
        num_constraints = len(constraints)

        A = np.zeros((num_constraints, num_vars), dtype=float)
        b = np.zeros(num_constraints, dtype=float)
        constraint_types = []

        for i, constr in enumerate(constraints):
            if len(constr["coefficients"]) != num_vars:
                raise ValueError(f"Constraint '{constr.get('name', i)}' has incorrect number of coefficients.")
            A[i, :] = constr["coefficients"]
            b[i] = constr["rhs"]
            constraint_types.append(constr["type"])
        
        return {
            "c": c,
            "A": A,
            "b": b,
            "constraint_types": constraint_types,
            "objective_type": problem_data["objective"]["type"],
            "variable_names": variable_names
        }
    except Exception as e:
        # logger.error(f"Error converting problem to matrix form: {e}")
        print(f"Error converting problem to matrix form: {e}")
        return None

if __name__ == '__main__':
    sample_text_problem = """
    Maximize
    obj: 3x1 + 2x2
    Subject to:
    c1: 1x1 + 1x2 <= 4
    c2: 2x1 + 1x2 <= 5
    Bounds:
    x1 >= 0
    x2 >= 0
    """
    # parsed_data, logs = parse_lp_problem_from_text(sample_text_problem)
    # for log in logs:
    #     print(log)
    # if parsed_data:
    #     print("\nParsed Data:", parsed_data)

    #     matrix_form = convert_problem_to_matrix_form(parsed_data) # Sẽ lỗi vì parsed_data là None
    #     if matrix_form:
    #         print("\nMatrix Form:")
    #         for key, value in matrix_form.items():
    #             print(f"{key}:\n{value}")

    sample_problem_dict = {
        "objective": {"type": "maximize", "coefficients": [3, 2]},
        "variables": ["x1", "x2"],
        "constraints": [
            {"name": "c1", "coefficients": [1, 1], "type": "<=", "rhs": 4},
            {"name": "c2", "coefficients": [2, 1], "type": "<=", "rhs": 5},
        ]
    }
    print("\n--- Converting Dict to Matrix Form ---")
    matrix_data = convert_problem_to_matrix_form(sample_problem_dict)
    if matrix_data:
        for k, v in matrix_data.items():
            print(f"{k}:")
            if isinstance(v, np.ndarray):
                print(v)
            else:
                print(v)
    else:
        print("Failed to convert to matrix form.")