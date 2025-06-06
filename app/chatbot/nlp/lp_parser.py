# /app/chatbot/nlp/lp_parser.py
import re
import logging
from typing import Dict, List, Tuple, Optional, Any

from . import rule_templates

logger = logging.getLogger(__name__)

def _expand_non_negativity_constraints(text: str) -> str:
    """
    Mở rộng các ràng buộc không âm viết gọn, ví dụ: "x1, x2 >= 0"
    thành các ràng buộc riêng biệt "x1 >= 0; x2 >= 0;".
    Dấu chấm phẩy ở cuối là quan trọng để tách biệt.
    """
    # Regex tìm một danh sách biến được ngăn cách bởi dấu phẩy
    pattern = re.compile(
        # Group 1: Bắt các biến (ví dụ: "x1, x2, var_3")
        r'((?:[a-zA-Z_][a-zA-Z0-9_]*\s*,\s*)+[a-zA-Z_][a-zA-Z0-9_]*)'
        # Group 2: Bắt toán tử
        r'\s*(>=|≥)\s*'
        # Group 3: Bắt số 0
        r'(0\.?0*|0)',
        re.IGNORECASE
    )

    def expander(match):
        vars_part, op, rhs = match.groups()
        variables = [v.strip() for v in vars_part.split(',')]
        # Trả về một chuỗi các ràng buộc, mỗi cái kết thúc bằng dấu chấm phẩy
        return '; '.join(f"{var} {op} {rhs}" for var in variables) + ';'

    # Thay thế tất cả các mẫu tìm thấy
    return pattern.sub(expander, text)

def parse_expression_to_coeffs_map(expr_str: str) -> Tuple[Dict[str, float], List[str]]:
    """
    Phân tích một biểu thức toán học (ví dụ: "3x1 - 2.5x2") thành
    một map hệ số và danh sách các biến theo thứ tự.
    """
    coeffs_map: Dict[str, float] = {}
    
    # Chuẩn hóa biểu thức để luôn bắt đầu bằng một dấu
    processed_expr = expr_str.strip()
    if not processed_expr.startswith(('+', '-')):
        processed_expr = '+' + processed_expr

    # Regex để tìm các số hạng: [dấu][hệ số]? [dấu nhân]? [tên biến]
    # Ví dụ: +3x1, - 2.5*x2, +x3
    term_regex = re.compile(r'([+\-])\s*(\d+\.?\d*)?\s*\*?\s*([a-zA-Z_][a-zA-Z0-9_]*)')
    
    for match in term_regex.finditer(processed_expr):
        sign, coeff_str, var_name = match.groups()
        
        coeff = 1.0
        if coeff_str:
            coeff = float(coeff_str)
        
        if sign == '-':
            coeff *= -1
        
        coeffs_map[var_name] = coeffs_map.get(var_name, 0.0) + coeff

    return coeffs_map, sorted(list(coeffs_map.keys()))

def parse_lp_problem_from_string(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Phân tích toàn bộ chuỗi bài toán LP một cách mạnh mẽ bằng cách tìm kiếm
    các mẫu ràng buộc thay vì phụ thuộc vào dấu phân cách.
    """
    logs = [f"LpParser: Bắt đầu phân tích: '{text[:200]}...'"]
    
    try:
        clean_text = text.replace('\n', ' ').strip()
        
        # --- Tách mục tiêu và ràng buộc ---
        objective_match = re.search(rule_templates.LP_PATTERNS['objective_keywords'], clean_text, re.IGNORECASE)
        if not objective_match:
            logs.append("Lỗi: Không tìm thấy từ khóa mục tiêu (maximize/minimize).")
            return None, logs
            
        objective_type_str = objective_match.group(1).lower()
        
        constraints_part = ""
        objective_part = ""

        st_match = re.search(rule_templates.LP_PATTERNS['subject_to_keywords'], clean_text, re.IGNORECASE)
        
        if st_match:
            objective_part = clean_text[objective_match.end():st_match.start()].strip()
            constraints_part = clean_text[st_match.end():].strip()
        else: # Nếu không có "s.t."
            # Tìm vị trí bắt đầu của ràng buộc đầu tiên
            temp_part = clean_text[objective_match.end():]
            op_match = re.search(r'(<=|>=|==|=|<|>|≤|≥)', temp_part)
            if op_match:
                # Tách phần mục tiêu và ràng buộc dựa vào vị trí của toán tử đầu tiên
                # Điều này giả định mục tiêu không chứa các toán tử này
                objective_part = temp_part[:op_match.start()]
                # Tìm ngược lại để lấy toàn bộ vế trái
                lhs_start_match = re.search(r'\s([+\-]?\s*\d*\.?\d*\s*\*?\s*[a-zA-Z_])', objective_part)
                if lhs_start_match:
                    objective_part = objective_part[:lhs_start_match.start()]
                    constraints_part = temp_part[lhs_start_match.start():]

            else: # Không tìm thấy ràng buộc nào
                objective_part = clean_text[objective_match.end():].strip()

        objective_expr = re.sub(r"^[a-zA-Z0-9\s_]+\s*=\s*", "", objective_part, flags=re.IGNORECASE).strip()
        obj_coeffs_map, obj_vars = parse_expression_to_coeffs_map(objective_expr)
        
        if not obj_coeffs_map:
            logs.append(f"Lỗi: Không phân tích được biểu thức mục tiêu: '{objective_expr}'")
            return None, logs

        problem_data = {
            "objective_type": "maximize" if "max" in objective_type_str else "minimize",
            "objective_coeffs_map": obj_coeffs_map,
            "objective_variables_ordered": obj_vars,
            "constraints": []
        }
        logs.append(f"Đã phân tích hàm mục tiêu: {problem_data['objective_type']}")

        # --- Phân tích ràng buộc (Logic Mới Mạnh Mẽ Hơn) ---
        if constraints_part:
            processed_constraints_str = _expand_non_negativity_constraints(constraints_part)
            
            # Regex để tìm tất cả các mẫu ràng buộc trong chuỗi
            constraint_finder_pattern = re.compile(
                r'([^<>=≤≥,;]+)'  # Group 1: Vế trái (bất cứ thứ gì không phải là toán tử hoặc dấu phân cách)
                r'\s*(<=|>=|==|=|<|>|≤|≥)\s*' # Group 2: Toán tử
                r'(-?\d+\.?\d*)' # Group 3: Vế phải (một con số)
            )
            
            matches = constraint_finder_pattern.finditer(processed_constraints_str)

            for i, match in enumerate(matches):
                lhs_str, op, rhs_str = [m.strip() for m in match.groups()]
                
                try:
                    rhs_val = float(rhs_str)
                    lhs_coeffs, _ = parse_expression_to_coeffs_map(lhs_str)
                    if not lhs_coeffs:
                        logs.append(f"Cảnh báo: Không tìm thấy biến ở vế trái của ràng buộc '{lhs_str}', bỏ qua.")
                        continue

                    problem_data["constraints"].append({
                        "name": f"c{i+1}", 
                        "coeffs_map": lhs_coeffs,
                        "operator": op.replace("=", "==").replace("<==", "<=").replace(">==", ">="), 
                        "rhs": rhs_val
                    })
                    logs.append(f"Đã phân tích ràng buộc: {lhs_str} {op} {rhs_str}")
                except (ValueError, TypeError):
                    logs.append(f"Cảnh báo: Vế phải của ràng buộc không phải là số trong '{line}', bỏ qua.")
                    continue
        
        logs.append(f"Phân tích hoàn tất. Tìm thấy {len(problem_data['constraints'])} ràng buộc.")
        return problem_data, logs

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong lp_parser: {e}", exc_info=True)
        logs.append(f"Lỗi nghiêm trọng không mong muốn: {e}")
        return None, logs
