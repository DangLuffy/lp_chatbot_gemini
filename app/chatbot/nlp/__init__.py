# /app/chatbot/nlp/__init__.py

# Đánh dấu thư mục 'nlp' là một package của Python.
# Đồng thời, import các thành phần chính để dễ dàng truy cập từ bên ngoài.

from .lp_parser import parse_lp_problem_from_string, parse_expression_to_coeffs_map
from .nlp_parser import NlpParser
from .nlp_gpt_parser import NlpGptParser # Giả sử bạn sẽ triển khai sau
# rule_templates và gpt_prompts thường được sử dụng nội bộ trong các parser.

# Bạn có thể khởi tạo một instance của NlpParser ở đây nếu muốn dùng chung
# Hoặc để DialogManager tự khởi tạo.
# nlp_parser_instance = NlpParser()
