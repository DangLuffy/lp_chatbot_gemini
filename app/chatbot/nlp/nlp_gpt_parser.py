# /app/chatbot/nlp/nlp_gpt_parser.py
import sys
import os
import logging
import json
import re
from typing import Dict, Any, Optional, List

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Gemini library
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("Thư viện google-generativeai chưa được cài đặt.")

# Import prompts
from app.chatbot.nlp.gpt_prompts import (
    PARSE_USER_REQUEST_TO_LP_PROMPT,
    EXPLAIN_LP_CONCEPT_PROMPT,
    GENERAL_CONVERSATION_PROMPT,
    CONVERT_STORY_TO_LP_PROMPT,  # <-- ✨ New
    SUGGEST_IMPROVEMENTS_PROMPT # <-- ✨ New
)

logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

class NlpGptParser:
    def __init__(self, api_key: Optional[str] = GEMINI_API_KEY, model_name: str = "gemini-1.5-flash-latest"):
        self.logs: List[str] = []
        self.api_key = api_key
        self.model_name = model_name
        self.model: Optional[genai.GenerativeModel] = None

        if not genai:
            self._log("Lỗi: Thư viện 'google-generativeai' không được tìm thấy.")
            return

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self._log(f"Đã cấu hình API Gemini thành công với model '{self.model_name}'.")
            except Exception as e:
                self._log(f"Lỗi khi cấu hình API Gemini: {e}")
                self.model = None
        else:
            self._log("Cảnh báo: API Key cho Gemini không được cung cấp.")

    def _log(self, message: str):
        self.logs.append(message)
        logger.info(f"NlpGptParser: {message}")

    async def _call_llm_api(self, prompt: str) -> Optional[str]:
        self.logs.clear()
        if not self.model:
            self._log("Lỗi: Model LLM (Gemini) chưa được khởi tạo.")
            return None

        self._log(f"Đang gửi prompt tới LLM '{self.model_name}': '{prompt[:200]}...'")
        try:
            response = await self.model.generate_content_async(prompt)
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                self._log("LLM đã tạo văn bản.")
                return generated_text.strip()
            else:
                self._log(f"Cấu trúc phản hồi LLM không như mong đợi: {response}")
                return None
        except Exception as e:
            self._log(f"Lỗi khi gọi API LLM: {e}")
            return None
            
    async def _call_llm_for_json(self, prompt: str) -> Optional[Dict]:
        response_str = await self._call_llm_api(prompt)
        if not response_str:
            return None
        
        try:
            # Trích xuất JSON từ khối mã markdown
            json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_str, re.IGNORECASE)
            json_str = json_match.group(1) if json_match else response_str
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            self._log(f"Lỗi khi giải mã JSON từ phản hồi LLM: {e}. Phản hồi là: {response_str}")
            return None

    async def handle_general_conversation(self, user_message: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        prompt = GENERAL_CONVERSATION_PROMPT.format(chat_history=history_str, user_message=user_message)
        return await self._call_llm_api(prompt)

    async def parse_user_request_to_lp_structure(self, user_message: str) -> Optional[Dict[str, Any]]:
        prompt = PARSE_USER_REQUEST_TO_LP_PROMPT.format(user_message=user_message)
        return await self._call_llm_for_json(prompt)

    async def explain_lp_concept(self, concept_name: str) -> Optional[str]:
        prompt = EXPLAIN_LP_CONCEPT_PROMPT.format(concept_name=concept_name)
        return await self._call_llm_api(prompt)
        
    # --- ✨ HÀM MỚI ---
    async def convert_story_to_lp(self, user_story: str) -> Optional[Dict[str, Any]]:
        """Sử dụng LLM để chuyển câu chuyện thành bài toán LP có cấu trúc JSON."""
        prompt = CONVERT_STORY_TO_LP_PROMPT.format(user_story=user_story)
        return await self._call_llm_for_json(prompt)
        
    # --- ✨ HÀM MỚI ---
    async def suggest_improvements(self, problem_context: Dict[str, Any]) -> Optional[str]:
        """Sử dụng LLM để đưa ra gợi ý cải thiện dựa trên kết quả."""
        # Trích xuất dữ liệu từ context để đưa vào prompt
        problem_def = problem_context.get("problem_definition", {})
        solution = problem_context.get("solution", {})
        
        prompt = SUGGEST_IMPROVEMENTS_PROMPT.format(
            objective_type=problem_def.get("objective_type", "N/A"),
            objective_expression=problem_def.get("objective_expression_str", "N/A"),
            constraints_list_str="\n".join(f"- {c}" for c in problem_def.get("constraints_str", ["N/A"])),
            status=solution.get("status", "N/A"),
            objective_value=solution.get("objective_value", "N/A"),
            variables=json.dumps(solution.get("variables", {})),
            solver_logs="\n".join(problem_context.get("logs", []))
        )
        return await self._call_llm_api(prompt)
# /app/chatbot/nlp/nlp_gpt_parser.py
import sys
import os
import logging
import json
import re
from typing import Dict, Any, Optional, List

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Gemini library
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("Thư viện google-generativeai chưa được cài đặt.")

# Import prompts
from app.chatbot.nlp.gpt_prompts import (
    PARSE_USER_REQUEST_TO_LP_PROMPT,
    EXPLAIN_LP_CONCEPT_PROMPT,
    GENERAL_CONVERSATION_PROMPT,
    CONVERT_STORY_TO_LP_PROMPT,
    SUGGEST_IMPROVEMENTS_PROMPT,
    EXPLAIN_SIMPLEX_STEP_PROMPT # ✨ Import prompt mới
)

logger = logging.getLogger(__name__)

class NlpGptParser:
    def __init__(self, api_key: Optional[str] = os.getenv("GOOGLE_API_KEY"), model_name: str = "gemini-1.5-flash-latest"):
        self.logs: List[str] = []
        self.api_key = api_key
        self.model_name = model_name
        self.model: Optional[genai.GenerativeModel] = None

        if not genai:
            self._log("Lỗi: Thư viện 'google-generativeai' không được tìm thấy.")
            return

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self._log(f"Đã cấu hình API Gemini thành công với model '{self.model_name}'.")
            except Exception as e:
                self._log(f"Lỗi khi cấu hình API Gemini: {e}")
                self.model = None
        else:
            self._log("Cảnh báo: API Key cho Gemini không được cung cấp.")

    def _log(self, message: str):
        self.logs.append(message)
        logger.info(f"NlpGptParser: {message}")

    async def _call_llm_api(self, prompt: str) -> Optional[str]:
        self.logs.clear()
        if not self.model:
            self._log("Lỗi: Model LLM (Gemini) chưa được khởi tạo.")
            return None

        self._log(f"Đang gửi prompt tới LLM '{self.model_name}': '{prompt[:200]}...'")
        try:
            response = await self.model.generate_content_async(prompt)
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                self._log("LLM đã tạo văn bản.")
                return generated_text.strip()
            else:
                self._log(f"Cấu trúc phản hồi LLM không như mong đợi: {response}")
                return None
        except Exception as e:
            self._log(f"Lỗi khi gọi API LLM: {e}")
            return None
            
    async def _call_llm_for_json(self, prompt: str) -> Optional[Dict]:
        response_str = await self._call_llm_api(prompt)
        if not response_str:
            return None
        
        try:
            # Trích xuất JSON từ khối mã markdown
            json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_str, re.IGNORECASE)
            json_str = json_match.group(1) if json_match else response_str
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            self._log(f"Lỗi khi giải mã JSON từ phản hồi LLM: {e}. Phản hồi là: {response_str}")
            return None

    async def handle_general_conversation(self, user_message: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        prompt = GENERAL_CONVERSATION_PROMPT.format(chat_history=history_str, user_message=user_message)
        return await self._call_llm_api(prompt)

    async def parse_user_request_to_lp_structure(self, user_message: str) -> Optional[Dict[str, Any]]:
        prompt = PARSE_USER_REQUEST_TO_LP_PROMPT.format(user_message=user_message)
        return await self._call_llm_for_json(prompt)

    async def explain_lp_concept(self, concept_name: str) -> Optional[str]:
        prompt = EXPLAIN_LP_CONCEPT_PROMPT.format(concept_name=concept_name)
        return await self._call_llm_api(prompt)
        
    async def convert_story_to_lp(self, user_story: str) -> Optional[Dict[str, Any]]:
        prompt = CONVERT_STORY_TO_LP_PROMPT.format(user_story=user_story)
        return await self._call_llm_for_json(prompt)
        
    async def suggest_improvements(self, problem_context: Dict[str, Any]) -> Optional[str]:
        problem_def = problem_context.get("problem_definition", {})
        solution = problem_context.get("solution", {})
        
        prompt = SUGGEST_IMPROVEMENTS_PROMPT.format(
            objective_type=problem_def.get("objective_type", "N/A"),
            objective_expression=problem_def.get("objective_expression_str", "N/A"),
            constraints_list_str="\n".join(f"- {c}" for c in problem_def.get("constraints_str", ["N/A"])),
            status=solution.get("status", "N/A"),
            objective_value=solution.get("objective_value", "N/A"),
            variables=json.dumps(solution.get("variables", {})),
            solver_logs="\n".join(problem_context.get("logs", []))
        )
        return await self._call_llm_api(prompt)

    # --- ✨ HÀM MỚI ĐỂ GIẢI THÍCH BƯỚC GIẢI ---
    async def explain_simplex_step(self, step_log_chunk: str) -> Optional[str]:
        """Sử dụng LLM để diễn giải một đoạn log của bước giải Simplex."""
        if not self.model:
            self._log("Lỗi: Model LLM chưa được khởi tạo để giải thích bước giải.")
            return "Xin lỗi, tôi không thể phân tích ngay lúc này do lỗi kết nối AI."
            
        prompt = EXPLAIN_SIMPLEX_STEP_PROMPT.format(step_log_chunk=step_log_chunk)
        return await self._call_llm_api(prompt)
