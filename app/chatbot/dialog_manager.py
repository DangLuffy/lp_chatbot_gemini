# /app/chatbot/dialog_manager.py
import logging
import json
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from .nlp import (
    parse_lp_problem_from_string,
    NlpParser,
    NlpGptParser
)
from app.solver.dispatcher import dispatch_solver

logger = logging.getLogger(__name__)

class DialogManager:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.logs: List[str] = []
        self.rule_based_nlp = NlpParser()
        self.lp_formula_parser = parse_lp_problem_from_string
        self.gpt_nlp = NlpGptParser()
        self.sample_problems = self._load_sample_problems()
        self.reset_state()

    def _load_sample_problems(self) -> List[Dict[str, Any]]:
        """Tải các bài toán mẫu từ tệp JSON."""
        try:
            # Đường dẫn đến tệp JSON trong cùng thư mục nlp
            json_path = Path(__file__).resolve().parent / "nlp" / "sample_problems.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                problems = json.load(f)
                self._log(f"Đã tải thành công {len(problems)} bài toán mẫu.")
                return problems
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self._log(f"Lỗi: Không thể tải tệp bài toán mẫu. {e}")
            return []

    def _log(self, message: str):
        entry = f"DM ({self.user_id}): {message}"
        self.logs.append(entry)
        logger.info(entry)

    def reset_state(self):
        """Reset trạng thái hội thoại về ban đầu."""
        self.state: Dict[str, Any] = {
            "history": [],
            "current_problem_definition": {}, # Dạng coeffs_map
            "last_solution_context": None, # Chứa cả problem, solution, logs
            "expectation": None, # Trạng thái chờ đợi hành động từ user
            "pending_action": None # Hành động đang chờ xác nhận
        }
        self._log("Trạng thái hội thoại đã được reset.")

    def _is_problem_defined(self) -> bool:
        """Kiểm tra xem đã có bài toán hoàn chỉnh trong bộ nhớ chưa."""
        p = self.state.get("current_problem_definition", {})
        return bool(p and p.get("objective_type") and p.get("objective_coeffs_map"))

    def _convert_internal_to_solver_format(self, internal_def: Dict) -> Optional[Dict[str, Any]]:
        """Chuyển đổi từ định dạng coeffs_map nội bộ sang Định dạng A cho solver."""
        try:
            all_vars = set(internal_def.get("objective_variables_ordered", []))
            for c in internal_def.get("constraints", []):
                all_vars.update(c.get("coeffs_map", {}).keys())
            ordered_vars = sorted(list(all_vars))

            return {
                "objective": internal_def["objective_type"],
                "coeffs": [internal_def["objective_coeffs_map"].get(v, 0.0) for v in ordered_vars],
                "variables_names_for_title_only": ordered_vars,
                "constraints": [{
                    "name": c.get("name", f"c{i+1}"),
                    "lhs": [c["coeffs_map"].get(v, 0.0) for v in ordered_vars],
                    "op": c["operator"], "rhs": c["rhs"]
                } for i, c in enumerate(internal_def.get("constraints", []))]
            }
        except Exception as e:
            self._log(f"Lỗi khi chuyển đổi định dạng cho solver: {e}")
            return None

    def _map_solver_name(self, user_input: str) -> str:
        """Chuẩn hóa tên solver từ input của người dùng."""
        user_input = user_input.lower()
        if "hìn" in user_input or "geo" in user_input: return "geometric"
        if "bland" in user_input: return "simplex_bland"
        if "đơn hình" in user_input or "simple" in user_input: return "simple_dictionary"
        if "pulp" in user_input: return "pulp_cbc"
        if "aux" in user_input: return "auxiliary"
        return "pulp_cbc"

    def _extract_log_chunk_for_step(self, step_number: int) -> Optional[str]:
        """Trích xuất khối log cho một iteration cụ thể."""
        context = self.state.get("last_solution_context")
        if not context: return None
        logs = context.get("logs", [])
        if not logs: return None

        log_str = "\n".join(logs)
        pattern = re.compile(rf"--- Iteration {step_number}[^\n]* ---\n(.*?)(?=\n--- Iteration|\Z)", re.DOTALL)
        match = pattern.search(log_str)
        
        if match:
            self._log(f"Đã tìm thấy log cho bước {step_number}.")
            return match.group(0).strip()
        else:
            self._log(f"Không tìm thấy log cho bước {step_number}.")
            return None

    # --- Các hàm xử lý (Handler Functions) ---

    async def _handle_intent_request_step_explanation(self, entities: Dict):
        """Xử lý yêu cầu giải thích một bước giải."""
        if not self.state.get("last_solution_context"):
            return self._finalize_response({"text_response": "Mình chưa có lời giải nào trong bộ nhớ để giải thích. Bạn hãy giải một bài toán trước nhé."})

        try:
            step_number = int(entities.get("step_number", "0"))
            if step_number <= 0: raise ValueError
        except (ValueError, TypeError):
            return self._finalize_response({"text_response": "Mình không hiểu bạn muốn giải thích bước nào. Vui lòng nói rõ, ví dụ: 'giải thích bước 2'."})

        log_chunk = self._extract_log_chunk_for_step(step_number)
        
        if not log_chunk:
            return self._finalize_response({"text_response": f"Mình không tìm thấy thông tin chi tiết cho bước {step_number} trong lần giải vừa rồi. Có thể bài toán được giải bằng phương pháp không có bước lặp, hoặc đã kết thúc sớm hơn."})
            
        explanation = await self.gpt_nlp.explain_simplex_step(log_chunk)
        
        return self._finalize_response({
            "text_response": explanation or "Xin lỗi, mình chưa thể giải thích bước này.",
            "allow_html": True,
            "suggestions": [f"Giải thích bước {step_number + 1}", "Trở về bài toán"]
        })

    async def _handle_intent_request_specific_solver(self, entities: Dict):
        """Xử lý yêu cầu giải lại bài toán với một solver cụ thể."""
        if not self._is_problem_defined():
            return self._finalize_response({"text_response": "Mình chưa có bài toán nào để giải lại. Bạn vui lòng cung cấp một bài toán trước nhé."})

        solver_name_entity = entities.get("solver_name", "")
        solver_to_use = self._map_solver_name(solver_name_entity)
        self._log(f"Chuẩn bị giải lại bằng solver: '{solver_to_use}'")
        
        # Đặt hành động chờ xác nhận
        self.state["pending_action"] = {"action": "solve", "solver": solver_to_use}
        self.state["expectation"] = "awaiting_confirmation"

        # Tạo câu hỏi xác nhận thân thiện
        problem_def = self.state['current_problem_definition']
        obj_expr = self._format_coeffs_map_for_display(problem_def.get('objective_coeffs_map', {}))
        
        confirmation_text = f"Được chứ! Bạn muốn mình giải lại bài toán hiện tại (<b>{obj_expr}</b>) bằng phương pháp <b>{solver_to_use}</b> phải không?"
        
        return self._finalize_response({
            "text_response": confirmation_text,
            "allow_html": True,
            "suggestions": ["Đúng vậy", "Thôi, để sau"]
        })

    def _handle_list_sample_problems(self):
        """Liệt kê các bài toán mẫu cho người dùng chọn."""
        if not self.sample_problems:
            return self._finalize_response({"text_response": "Xin lỗi, mình chưa có sẵn bài toán mẫu nào cả."})
        
        response_lines = ["Mình có một vài bài toán mẫu đây, bạn muốn thử bài nào?"]
        suggestions = []
        for i, problem in enumerate(self.sample_problems):
            response_lines.append(f"<b>{i+1}. {problem['name']}</b>: {problem['story']}")
            suggestions.append(f"Chọn bài toán {i+1}")
            
        self.state["expectation"] = "awaiting_sample_choice"
        return self._finalize_response({
            "text_response": "<br>".join(response_lines),
            "allow_html": True,
            "suggestions": suggestions
        })

    async def _handle_sample_choice(self, user_message: str):
        """Xử lý khi người dùng chọn một bài toán mẫu."""
        try:
            choice_match = re.search(r'\d+', user_message)
            if not choice_match: raise ValueError
            choice_index = int(choice_match.group(0)) - 1
            
            if 0 <= choice_index < len(self.sample_problems):
                chosen_problem = self.sample_problems[choice_index]
                self._log(f"Người dùng đã chọn bài toán mẫu: {chosen_problem['name']}")
                # Parse bài toán từ chuỗi và giải
                parsed_lp, _ = self.lp_formula_parser(chosen_problem['full_problem_string'])
                self.state['current_problem_definition'] = parsed_lp
                self.state['expectation'] = None # Xóa trạng thái chờ
                return await self._solve_current_problem("pulp_cbc") # Giải bằng solver mặc định
            else:
                raise IndexError
        except (ValueError, IndexError):
            return self._finalize_response({"text_response": "Lựa chọn không hợp lệ. Bạn vui lòng chọn lại từ danh sách nhé."})

    async def _solve_current_problem(self, solver_name: str, preamble: Optional[str] = None):
        """Hàm tổng hợp để giải bài toán hiện tại trong state."""
        internal_def = self.state["current_problem_definition"]
        solver_format = self._convert_internal_to_solver_format(internal_def)
        if not solver_format:
            return self._finalize_response({"text_response": "Rất tiếc, có lỗi khi chuẩn bị dữ liệu để giải.", "allow_html": False})

        self._log(f"Bắt đầu giải bằng solver '{solver_name}'...")
        solution, logs = dispatch_solver(solver_format, solver_name=solver_name)
        
        # Lưu lại toàn bộ ngữ cảnh của lần giải này
        context = {"problem_definition": internal_def, "solution": solution, "logs": logs}
        self.state["last_solution_context"] = context

        # Định dạng câu trả lời
        response_parts = []
        if preamble: response_parts.append(f"<p>{preamble}</p>")
        response_parts.append(self._format_problem_summary(internal_def))
        response_parts.append(self._format_solution_response(solution, solver_format, solver_name))
        
        # Tạo gợi ý
        suggestions = ["Bắt đầu bài toán mới"]
        if solution and solution.get("status") == "Optimal":
            if solver_name not in ["geometric"]:
                 suggestions.insert(0, "Giải thích bước 1")
            if len(solver_format.get("variables_names_for_title_only", [])) == 2 and solver_name != "geometric":
                 suggestions.insert(0, "Giải bằng hình học")
            elif len(solver_format.get("variables_names_for_title_only", [])) > 2 and solver_name != "simple_dictionary":
                 suggestions.insert(0, "Giải bằng đơn hình")

        return self._finalize_response({
            "text_response": "".join(response_parts),
            "problem_context": context,
            "allow_html": True,
            "suggestions": suggestions
        })

    async def handle_message(self, user_message: str) -> Dict[str, Any]:
        """Hàm chính điều phối luồng hội thoại."""
        self._log(f"Đang xử lý tin nhắn: '{user_message}' (Trạng thái chờ: {self.state['expectation']})")
        self.state["history"].append({"role": "user", "content": user_message})

        # Ưu tiên 1: Xử lý theo trạng thái chờ mong đợi
        if self.state["expectation"] == "awaiting_confirmation":
            self.state["expectation"] = None # Reset trạng thái chờ
            if "không" in user_message.lower() or "thôi" in user_message.lower():
                self.state["pending_action"] = None
                return self._finalize_response({"text_response": "Được rồi, nếu bạn cần gì khác cứ nói nhé!", "suggestions": ["Bắt đầu bài toán mới"]})
            else: # Mặc định là đồng ý
                pending = self.state.pop("pending_action")
                if pending and pending['action'] == 'solve':
                    return await self._solve_current_problem(pending['solver'])

        if self.state["expectation"] == "awaiting_sample_choice":
            return await self._handle_sample_choice(user_message)

        # Ưu tiên 2: Các lệnh đặc biệt
        if user_message.lower() in ["bắt đầu lại", "reset", "làm mới", "bài toán mới"]:
             self.reset_state()
             return self._finalize_response({"text_response": "Đã làm mới! Mình có thể giúp gì cho bạn tiếp theo?", "suggestions": ['Giải bài toán mẫu', 'Kể một câu chuyện bài toán', 'Biến bù là gì?']})
        
        if "bài toán mẫu" in user_message.lower():
             return self._handle_list_sample_problems()
        
        if "câu chuyện" in user_message.lower():
             self.state['expectation'] = 'awaiting_story'
             return self._finalize_response({"text_response": "Tuyệt vời! Hãy kể cho mình nghe vấn đề của bạn bằng ngôn ngữ tự nhiên nhé. Mình sẽ cố gắng chuyển nó thành một bài toán LP."})
        
        if self.state['expectation'] == 'awaiting_story':
             # Xử lý câu chuyện... (Logic này có thể được thêm vào sau)
             pass

        # Ưu tiên 3: Thử parse như một bài toán đầy đủ
        parsed_lp, parse_logs = self.lp_formula_parser(user_message)
        if parsed_lp and self._is_problem_defined(parsed_lp):
            self._log(f"Đã phân tích thành công bài toán từ chuỗi. Logs: {parse_logs}")
            self.state["current_problem_definition"] = parsed_lp
            return await self._solve_current_problem("pulp_cbc")

        # Ưu tiên 4: Phân tích ý định từ câu nói
        nlp_result = self.rule_based_nlp.parse_intent_and_entities(user_message)
        intent = nlp_result.get("intent", "unknown")
        entities = nlp_result.get("entities", {})

        if intent == "request_step_explanation":
            return await self._handle_intent_request_step_explanation(entities)
        
        if intent == "request_specific_solver":
            return await self._handle_intent_request_specific_solver(entities)

        if intent == "request_theoretical_concept":
            concept = entities.get("concept_name", "khái niệm đó")
            explanation = await self.gpt_nlp.explain_lp_concept(concept) or f"Mình chưa có thông tin về '{concept}'."
            return self._finalize_response({"text_response": explanation, "allow_html": True, "suggestions": ["Quy tắc Bland là gì?", "Biến nhân tạo là gì?"]})
        
        # Mặc định: Dùng LLM để trò chuyện
        if self.gpt_nlp.model:
            response_text = await self.gpt_nlp.handle_general_conversation(user_message, self.state["history"]) or "Xin lỗi, mình chưa hiểu ý bạn. Bạn có thể nói rõ hơn được không?"
            return self._finalize_response({"text_response": response_text, "suggestions": ["Giải bài toán mẫu"]})
        else:
            return self._finalize_response({"text_response": "Chào bạn, mình là trợ lý Quy hoạch tuyến tính. Mình có thể giúp gì cho bạn?", "suggestions": ["Giải bài toán mẫu"]})

    # --- Các hàm định dạng (Formatting Functions) ---
    def _format_problem_summary(self, internal_def: Dict) -> str:
        """Tạo bản tóm tắt bài toán bằng HTML."""
        obj_type = internal_def.get('objective_type', 'N/A').capitalize()
        obj_expr = self._format_coeffs_map_for_display(internal_def.get('objective_coeffs_map', {}))
        summary_html = f"<b>Bài toán của bạn:</b><ul><li><b>Mục tiêu:</b> {obj_type} Z = {obj_expr}</li>"
        
        if internal_def.get("constraints"):
            summary_html += "<li><b>Các ràng buộc:</b><ul style='padding-left: 20px;'>"
            for c in internal_def["constraints"]:
                lhs = self._format_coeffs_map_for_display(c['coeffs_map'])
                op = c.get('operator', '=').replace('==', '=')
                rhs = c.get('rhs', 0)
                summary_html += f"<li>{lhs} {op} {rhs}</li>"
            summary_html += "</ul></li>"
        summary_html += "</ul><hr style='margin: 12px 0; border-color: #e2e8f0;'>"
        return summary_html
        
    def _format_solution_response(self, solution: dict, problem_data: dict, solver_name: str) -> str:
        """Định dạng câu trả lời kết quả, có xử lý hình ảnh."""
        if not solution:
            return f"<p><b>Kết quả từ phương pháp '{solver_name}':</b></p><p>Rất tiếc, đã có lỗi xảy ra và bộ giải không trả về kết quả.</p>"

        status = solution.get("status", "Unknown")
        header = f"<p><b>Kết quả từ phương pháp '{solver_name}':</b></p>"
        body = ""
        
        if status == "Optimal":
            obj_val = solution.get("objective_value", 0)
            vars_map = solution.get("variables", {})
            vars_str = ", ".join([f"<b>{k}</b> = {v:.3f}" for k, v in vars_map.items() if not k.startswith('_')])
            body = f"🎉 <b>Lời giải tối ưu đã được tìm thấy!</b><br>• <b>Giá trị mục tiêu:</b> {obj_val:.3f}<br>• <b>Giá trị các biến:</b> {vars_str}"
        elif status == "Infeasible":
            body = "Trạng thái bài toán: <b>Vô nghiệm</b>. Các ràng buộc có thể đang mâu thuẫn với nhau."
        elif status == "Unbounded":
            body = "Trạng thái bài toán: <b>Không bị chặn</b>. Hàm mục tiêu có thể tiến tới vô cùng."
        else:
            body = f"Trạng thái bài toán: <b>{status}</b>. Không tìm thấy lời giải tối ưu."
            
        image_html = ""
        if solution.get("plot_image_base64"):
            img_src = solution["plot_image_base64"]
            image_html = f"<br><br><div style='text-align: center;'><img src='{img_src}' alt='Biểu đồ giải bằng hình học' style='max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);' /></div>"
            
        return header + body + image_html

    def _format_coeffs_map_for_display(self, coeffs_map: Dict[str, float]) -> str:
        if not coeffs_map: return "0"
        terms = []
        is_first = True
        for var, coeff in sorted(coeffs_map.items()):
            if abs(coeff) < 1e-9: continue
            
            sign = ""
            if not is_first:
                sign = " + " if coeff > 0 else " - "
            elif coeff < 0:
                sign = "-"

            abs_coeff = abs(coeff)
            coeff_part = str(round(abs_coeff, 2)) if abs(round(abs_coeff, 2) - 1.0) > 1e-9 or var == '' else ""
            
            terms.append(f"{sign}{coeff_part}{var}")
            is_first = False
        return "".join(terms).strip()

    def _finalize_response(self, response: Dict) -> Dict:
        """Đóng gói và trả về phản hồi cuối cùng."""
        response.setdefault("text_response", "Mình chưa rõ ý bạn, bạn có thể nói khác đi được không?")
        response.setdefault("suggestions", ["Giải bài toán mẫu", "Bắt đầu bài toán mới"])
        response.setdefault("allow_html", False)
        
        self.state["history"].append({"role": "assistant", "content": response["text_response"]})
        self.state["last_bot_message"] = response["text_response"]
        return response
