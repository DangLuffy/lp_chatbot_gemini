# /app/chatbot/nlp/nlp_parser.py
import re
import logging
from typing import Dict, List, Any, Optional

from . import rule_templates
from .lp_parser import parse_expression_to_coeffs_map

logger = logging.getLogger(__name__)

class NlpParser:
    """
    Xử lý NLP dựa trên quy tắc: nhận diện ý định (intent) và trích xuất
    thực thể (entities) từ ngôn ngữ tự nhiên của người dùng.
    """
    def __init__(self):
        self.logs: List[str] = []

    def _log(self, message: str):
        self.logs.append(message)
        logger.debug(f"NlpParser: {message}")

    def clear_logs(self):
        self.logs = []

    def parse_intent_and_entities(self, text: str) -> Dict[str, Any]:
        """
        Xác định ý định và các thực thể chính từ văn bản người dùng.

        Hàm này sẽ thử khớp văn bản với các mẫu (patterns) đã được định nghĩa
        trong `rule_templates.py` để tìm ra hành động mà người dùng muốn thực hiện.

        Args:
            text: Chuỗi văn bản từ người dùng.

        Returns:
            Một dictionary chứa 'intent' và 'entities' đã được phân tích.
        """
        self.clear_logs()
        text_lower = text.lower().strip()
        self._log(f"Bắt đầu phân tích ý định cho: '{text_lower}'")

        # Duyệt qua các mẫu intent đã định nghĩa để tìm sự trùng khớp
        for intent, patterns in rule_templates.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    self._log(f"Đã tìm thấy intent: '{intent}' với pattern: '{pattern.pattern}'")
                    
                    entities = {"original_text": text}
                    
                    # Trích xuất các thông tin cụ thể (entities) nếu có
                    if intent == "request_step_explanation":
                        # Tìm số bước trong câu nói
                        step_match = re.search(r'\d+', text)
                        if step_match:
                            entities['step_number'] = step_match.group(0)
                        else:
                             # Nếu người dùng chỉ nói "hiển thị cách giải" mà không có số
                             entities['step_number'] = '1' # Mặc định là bước 1

                    elif intent == "request_specific_solver":
                         # Trích xuất tên solver được yêu cầu
                         solver_match = re.search(rule_templates.ENTITY_PATTERNS['solver_name'], text_lower)
                         if solver_match:
                             entities['solver_name'] = solver_match.group(0)

                    elif intent == "request_theoretical_concept":
                         # Trích xuất tên khái niệm nếu có thể
                         if len(match.groups()) > 0:
                             entities['concept_name'] = match.group(match.lastindex).strip()

                    return {"intent": intent, "entities": entities}

        # Nếu không có intent nào khớp, trả về 'unknown'
        self._log("Không tìm thấy intent cụ thể nào, trả về 'unknown'.")
        return {"intent": "unknown", "entities": {"original_text": text}}

    def parse_objective_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        (Không còn được sử dụng chính) Phân tích một chuỗi chỉ chứa hàm mục tiêu.
        Logic này đã được tích hợp vào `lp_parser.py` để xử lý toàn diện hơn.
        """
        match = re.search(rule_templates.LP_PATTERNS["objective_keywords"], text, re.IGNORECASE)
        if not match:
            return None
        
        obj_type = "maximize" if "max" in match.group(1).lower() else "minimize"
        expr_str = text[match.end():].strip()
        expr_str = re.sub(r"^[a-zA-Z\s_]+[0-9_]*\s*=\s*", "", expr_str, flags=re.IGNORECASE).strip()
        
        coeffs_map, variables_ordered = parse_expression_to_coeffs_map(expr_str)
        if not coeffs_map:
            return None
            
        return {
            "objective_type": obj_type,
            "coeffs_map": coeffs_map,
            "variables_ordered": variables_ordered
        }

