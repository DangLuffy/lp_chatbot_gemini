# /app/chatbot/nlp.py

import re
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Các biểu thức chính quy (Regex) để phân tích văn bản
# Đây là những ví dụ rất cơ bản và cần được cải tiến nhiều.
PATTERNS = {
    "objective": re.compile(r"(tối đa hóa|tối thiểu hóa|maximize|minimize|max|min)\s*(.*?)(với điều kiện|subject to|s\.t\.|:)", re.IGNORECASE),
    "constraint": re.compile(r"(.*?)\s*(<=|>=|==|<|>)\s*(-?\d+\.?\d*)", re.IGNORECASE),
    "variable_coeff": re.compile(r"(-?\s*\d*\.?\d*)\s*\*?\s*([a-zA-Z]+\d*)")
}

def parse_expression(expr_str: str) -> Dict[str, float]:
    """
    Phân tích một biểu thức như "3x1 - 2x2 + x3" thành một dictionary hệ số.
    Ví dụ: {'x1': 3.0, 'x2': -2.0, 'x3': 1.0}
    """
    coeffs = {}
    # Thêm dấu '+' ở đầu để đảm bảo regex hoạt động với số hạng đầu tiên
    if not expr_str.strip().startswith(('+', '-')):
        expr_str = '+' + expr_str
    
    # Thay thế ' - ' bằng ' + -' để dễ dàng split
    expr_str = expr_str.replace(' - ', ' + -')
    terms = [term.strip() for term in expr_str.split('+') if term.strip()]
    
    for term in terms:
        term = term.replace(' ', '') # Loại bỏ khoảng trắng
        match = PATTERNS["variable_coeff"].fullmatch(term)
        if match:
            coeff_str, var_name = match.groups()
            
            if coeff_str == '' or coeff_str is None:
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
                
            coeffs[var_name] = coeffs.get(var_name, 0) + coeff
        else:
            logger.warning(f"Could not parse term: '{term}' in expression '{expr_str}'")
            
    return coeffs

def extract_lp_from_text(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Hàm chính để trích xuất một bài toán LP từ văn bản thô.

    Returns:
        Một tuple (problem_data, logs)
    """
    logs = [f"NLP: Starting to parse text: '{text}'"]
    problem_data = {
        "objective": {"type": "", "coefficients": []},
        "variables": [],
        "constraints": []
    }
    
    # 1. Trích xuất hàm mục tiêu
    obj_match = PATTERNS["objective"].search(text)
    if not obj_match:
        logs.append("NLP Error: Could not find objective function (e.g., 'maximize ... subject to ...').")
        return None, logs

    obj_type_str = obj_match.group(1).lower()
    obj_expr_str = obj_match.group(2).strip()
    
    if obj_type_str in ["tối đa hóa", "maximize", "max"]:
        problem_data["objective"]["type"] = "maximize"
    else:
        problem_data["objective"]["type"] = "minimize"
    logs.append(f"NLP: Found objective type: {problem_data['objective']['type']}")
    logs.append(f"NLP: Found objective expression: '{obj_expr_str}'")

    # Phân tích biểu thức mục tiêu để lấy biến và hệ số
    obj_coeffs_map = parse_expression(obj_expr_str)
    
    # Xác định danh sách các biến theo thứ tự
    # Sắp xếp theo tên biến để đảm bảo thứ tự nhất quán
    variable_names = sorted(obj_coeffs_map.keys())
    problem_data["variables"] = variable_names
    
    if not variable_names:
        logs.append("NLP Error: Could not identify any variables in the objective function.")
        return None, logs
    logs.append(f"NLP: Identified variables: {variable_names}")

    # Tạo vector hệ số mục tiêu theo đúng thứ tự
    problem_data["objective"]["coefficients"] = [obj_coeffs_map.get(var, 0) for var in variable_names]

    # 2. Trích xuất các ràng buộc
    # Lấy phần văn bản sau "subject to"
    constraints_text_start = obj_match.end(3)
    constraints_text = text[constraints_text_start:].strip()
    
    # Chia các ràng buộc theo dòng hoặc dấu phẩy/chấm phẩy
    constraint_lines = re.split(r'[\n,;]', constraints_text)
    
    for i, line in enumerate(constraint_lines):
        line = line.strip()
        if not line:
            continue
            
        constr_match = PATTERNS["constraint"].search(line)
        if not constr_match:
            logs.append(f"NLP Warning: Could not parse constraint from line: '{line}'")
            continue
        
        lhs_expr_str = constr_match.group(1)
        constr_type = constr_match.group(2)
        rhs_val = float(constr_match.group(3))
        
        # Phân tích vế trái của ràng buộc
        lhs_coeffs_map = parse_expression(lhs_expr_str)
        
        # Tạo vector hệ số cho ràng buộc này, đảm bảo cùng thứ tự với biến
        constraint_coeffs = [lhs_coeffs_map.get(var, 0) for var in variable_names]
        
        constraint = {
            "name": f"c{i+1}",
            "coefficients": constraint_coeffs,
            "type": constr_type,
            "rhs": rhs_val
        }
        problem_data["constraints"].append(constraint)
        logs.append(f"NLP: Parsed constraint {constraint['name']}: {lhs_expr_str} {constr_type} {rhs_val}")

    if not problem_data["constraints"]:
        logs.append("NLP Error: No valid constraints were found.")
        return None, logs

    logs.append("NLP: Parsing completed successfully.")
    return problem_data, logs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    test_text_1 = """
    Maximize 3x1 + 2x2 
    subject to 
    x1 + x2 <= 4,
    2x1 + x2 <= 5;
    -x1 + 4x2 >= -2
    """
    
    test_text_2 = "tối thiểu hóa 10a + 15b với điều kiện: a+b==100, 2a-b>=-5"

    print("--- Testing Case 1 ---")
    parsed_data, logs = extract_lp_from_text(test_text_1)
    for log in logs:
        print(log)
    if parsed_data:
        import json
        print("Parsed Data:\n", json.dumps(parsed_data, indent=2))
        
    print("\n--- Testing Case 2 ---")
    parsed_data_2, logs_2 = extract_lp_from_text(test_text_2)
    for log in logs_2:
        print(log)
    if parsed_data_2:
        import json
        print("Parsed Data:\n", json.dumps(parsed_data_2, indent=2))
