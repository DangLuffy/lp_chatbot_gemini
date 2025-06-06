# /app/chatbot/nlp/rule_templates.py
import re

# Các biểu thức chính quy (Regex) để phân tích văn bản
LP_PATTERNS = {
    "objective_keywords": r"(tối đa hóa|tối thiểu hóa|maximize|minimize|max|min)",
    "subject_to_keywords": r"(với điều kiện|subject to|s\.t\.|st)",
    "variable_coeff": r"(-?\s*\d*\.?\d*)\s*\*?\s*([a-zA-Z_][a-zA-Z0-9_]*)",
    "constraint_operator": r"(<=|>=|==|=|<|>|≤|≥)",
    "number": r"(-?\d+\.?\d*|-?\.\d+)"
}

LP_PATTERNS["full_objective_and_constraints_split"] = re.compile(
    rf"^\s*{LP_PATTERNS['objective_keywords']}\s*(.*?)\s*(?:{LP_PATTERNS['subject_to_keywords']}(.*))?$",
    re.IGNORECASE | re.DOTALL
)

LP_PATTERNS["single_constraint_line"] = re.compile(
    r"^(?:[a-zA-Z0-9\s\(\)\._-]+:)?\s*(.*?)\s*" + \
    LP_PATTERNS["constraint_operator"] + \
    r"\s*" + LP_PATTERNS["number"] + r"\s*$",
    re.IGNORECASE
)

# Từ khóa và mẫu cho Intent Recognition
# Từ khóa và mẫu cho Intent Recognition
INTENT_PATTERNS = {
    "greet": [re.compile(r"^\s*(xin chào|chào bạn|hello|hi|chào)\s*$", re.IGNORECASE)],
    
    # --- ✨ Cải tiến: Tách biệt và ưu tiên các intent cụ thể ---
    "request_step_explanation": [
        re.compile(r".*(giải thích|phân tích|hiển thị)\s+(từng\s+)?(bước|vòng lặp|iteration|log|cách giải)\s*(\d*).*", re.IGNORECASE)
    ],
    
    # --- ✨ Cải tiến: Regex cụ thể hơn cho từng loại solver ---
    "request_specific_solver": [
        re.compile(r".*(dùng|giải bằng|sử dụng|thử).*(từ điển đơn hình|đơn hình|simplex|simple dictionary).*", re.IGNORECASE),
        re.compile(r".*(dùng|giải bằng|sử dụng|thử).*(bland).*", re.IGNORECASE),
        re.compile(r".*(dùng|giải bằng|sử dụng|thử).*(hình học|geometric).*", re.IGNORECASE),
        re.compile(r".*(dùng|giải bằng|sử dụng|thử).*(pulp|cbc).*", re.IGNORECASE),
        re.compile(r".*(dùng|giải bằng|sử dụng|thử).*(auxiliary).*", re.IGNORECASE),
    ],
    
    "define_objective": [re.compile(r"^(tối đa hóa|tối thiểu hóa|maximize|minimize|max|min)\s+.*", re.IGNORECASE)],
    "add_constraint": [
        re.compile(r"^(ràng buộc|điều kiện|dk|rb|constraint).*", re.IGNORECASE),
        re.compile(r".*" + r"(<=|>=|==|=|<|>|≤|≥)" + r".*", re.IGNORECASE)
    ],
    "request_solve": [re.compile(r".*(giải|solve|kết quả|tìm lời giải).*", re.IGNORECASE)],
    "request_theoretical_concept": [
        re.compile(r".*(là gì|nghĩa là gì|define|what is)\s*$", re.IGNORECASE),
        re.compile(r"^(cho tôi biết về|nói về|explain)\s+([a-zA-Z0-9\s_À-ỹ]+)$", re.IGNORECASE)
    ],
    "reset_conversation": [re.compile(r"^(reset|bắt đầu lại|làm mới|clear|bài toán mới)\s*$", re.IGNORECASE)],
}

# Mẫu để trích xuất các thực thể cơ bản
ENTITY_PATTERNS = {
    # --- ✨ CẬP NHẬT: Thêm "hình học", "bland", "auxiliary" vào danh sách ---
    "solver_name": r"(pulp_cbc|geometric|simple_dictionary|simplex_bland|auxiliary|hình học|pulp|bland)",
    "concept_name_lp": r"(quy tắc Bland|biến nhân tạo|simplex|đơn hình đối ngẫu)"
}

# Mẫu để trích xuất các thực thể cơ bản
ENTITY_PATTERNS = {
    "solver_name": r"(pulp_cbc|geometric|simple_dictionary|simplex_bland|auxiliary|hình học|đơn hình|pulp|bland)",
    "concept_name_lp": r"(quy tắc Bland|biến nhân tạo|simplex|đơn hình đối ngẫu)"
}

OBJECTIVE_TYPE_KEYWORDS = {
    "maximize": [r"tối đa hóa", r"maximize", r"max"],
    "minimize": [r"tối thiểu hóa", r"minimize", r"min"]
}
CONSTRAINT_INDICATORS = [
    r"^\s*[a-zA-Z_][a-zA-Z0-9_]*.*" + LP_PATTERNS["constraint_operator"],
    r"^\s*\d+.*" + LP_PATTERNS["constraint_operator"],
]
