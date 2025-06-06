# /app/chatbot/nlp/nlp_parser.py
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from .rule_templates import INTENT_PATTERNS, ENTITY_PATTERNS, OBJECTIVE_TYPE_KEYWORDS, CONSTRAINT_INDICATORS
from .lp_parser import parse_expression_to_coeffs_map # Để parse các phần nhỏ

logger = logging.getLogger(__name__)

class NlpParser:
    """
    Xử lý NLP dựa trên quy tắc: nhận diện ý định, trích xuất thực thể,
    và phân tích các phần của bài toán LP từ ngôn ngữ tự nhiên.
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
        Xác định ý định (intent) và các thực thể (entities) từ văn bản.
        """
        self.clear_logs()
        self._log(f"Analyzing text for intent/entities: '{text}'")
        text_lower = text.lower().strip()
        
        # 1. Kiểm tra các intent dựa trên từ khóa và regex
        for intent_name, patterns_list in INTENT_PATTERNS.items():
            for pattern_re in patterns_list:
                match = pattern_re.search(text_lower)

                if match:
                    entities: Dict[str, Any] = {"original_text": text}
                    
                    # Trích xuất các groups nếu regex có
                    try:
                        if intent_name == "request_specific_solver":
                            solver_match = re.search(ENTITY_PATTERNS['solver_name'], text_lower, re.IGNORECASE)
                            if solver_match:
                                entities["solver_name"] = solver_match.group(0).strip()

                        elif intent_name == "request_theoretical_concept":
                             entities["concept_name"] = match.group(match.lastindex).strip()

                        elif intent_name == "request_step_explanation":
                             entities["step_number"] = match.group(match.lastindex).strip()
                    
                    except (IndexError, AttributeError):
                         # Không có group hoặc không khớp, bỏ qua
                         pass

                    self._log(f"Intent detected: '{intent_name}', Entities: {entities}")
                    return {"intent": intent_name, "entities": entities, "logs": self.logs}

        # 2. Heuristic: Nếu không có intent rõ ràng, kiểm tra cấu trúc
        is_likely_objective = any(re.search(kw, text_lower) for kw_pattern in OBJECTIVE_TYPE_KEYWORDS.values() for kw in kw_pattern)
        is_likely_constraint = any(re.search(indicator, text_lower) for indicator in CONSTRAINT_INDICATORS)

        if is_likely_objective:
            self._log("Text looks like an objective function by keywords.")
            return {"intent": "define_objective", "entities": {"original_text": text}, "logs": self.logs}
        if is_likely_constraint:
            self._log("Text looks like a constraint by structure.")
            return {"intent": "add_constraint", "entities": {"original_text": text}, "logs": self.logs}

        self._log("No specific intent detected. Defaulting to 'unknown'.")
        return {"intent": "unknown", "entities": {"original_text": text}, "logs": self.logs}
