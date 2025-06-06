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
        self.rule_based_nlp = NlpParser()
        self.lp_formula_parser = parse_lp_problem_from_string
        self.gpt_nlp = NlpGptParser()
        self.sample_problems = self._load_sample_problems()
        self.reset_state()

    def _load_sample_problems(self) -> List[Dict[str, Any]]:
        try:
            json_path = Path(__file__).resolve().parent / "nlp" / "sample_problems.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                problems = json.load(f)
            self._log(f"Đã tải thành công {len(problems)} bài toán mẫu.")
            return problems
        except Exception as e:
            self._log(f"Lỗi: Không thể tải tệp bài toán mẫu. {e}")
            return []

    def _log(self, message: str):
        logger.info(f"DM ({self.user_id}): {message}")

    def reset_state(self):
        self.state: Dict[str, Any] = { "history": [], "current_problem_definition": {}, "last_solution_context": None, "expectation": None, "pending_action": None }
        self._log("Trạng thái hội thoại đã được reset.")

    def _is_problem_defined(self) -> bool:
        p = self.state.get("current_problem_definition", {})
        return bool(p and p.get("objective_type") and p.get("objective_coeffs_map"))

    def _convert_internal_to_solver_format(self, internal_def: Dict) -> Optional[Dict[str, Any]]:
        try:
            all_vars = sorted(list(set(internal_def.get("objective_variables_ordered", [])).union(*(c.get("coeffs_map", {}).keys() for c in internal_def.get("constraints", [])))))
            return {
                "objective": internal_def["objective_type"],
                "coeffs": [internal_def["objective_coeffs_map"].get(v, 0.0) for v in all_vars],
                "variables_names_for_title_only": all_vars,
                "constraints": [{ "name": c.get("name", f"c{i+1}"), "lhs": [c["coeffs_map"].get(v, 0.0) for v in all_vars], "op": c["operator"], "rhs": c["rhs"] } for i, c in enumerate(internal_def.get("constraints", []))]
            }
        except Exception as e:
            self._log(f"Lỗi khi chuyển đổi định dạng cho solver: {e}")
            return None

    def _map_solver_name(self, user_input: str, problem_definition: Dict) -> str:
        """
        ✨ Cải tiến: Chọn solver Đơn hình tốt nhất dựa trên cấu trúc bài toán.
        """
        user_input = user_input.lower()
        
        # Ưu tiên các yêu cầu rõ ràng cho các solver không phải Đơn hình
        if "hìn" in user_input or "geo" in user_input: return "geometric"
        if "pulp" in user_input: return "pulp_cbc"

        # Nếu người dùng yêu cầu một phương pháp Đơn hình cụ thể
        if "bland" in user_input: return "simplex_bland"
        if "aux" in user_input: return "auxiliary"

        # Nếu người dùng yêu cầu "đơn hình" chung chung
        if "đơn hình" in user_input or "simple" in user_input or "simplex" in user_input:
            has_complex_constraints = any(
                c.get("operator") in ["==", "=", ">=", "≥"]
                for c in problem_definition.get("constraints", [])
            )
            
            if has_complex_constraints:
                self._log("Bài toán có ràng buộc phức tạp (==, >=). Tự động chọn solver 'auxiliary' để đảm bảo tính chính xác.")
                return "auxiliary"
            else:
                self._log("Bài toán chỉ có ràng buộc <=. Chọn solver 'simple_dictionary'.")
                return "simple_dictionary"

        return "pulp_cbc" # Mặc định cuối cùng

    def _extract_log_chunk_for_step(self, step_number: int) -> Optional[str]:
        context = self.state.get("last_solution_context")
        if not context or not context.get("logs"): return None
        log_str = "\n".join(context["logs"])
        pattern = re.compile(rf"--- Iteration {step_number}[^\n]* ---\n(.*?)(?=\n--- Iteration|\Z)", re.DOTALL)
        match = pattern.search(log_str)
        if match:
            return match.group(0).strip()
        return None

    async def _handle_intent_request_step_explanation(self, entities: Dict):
        step_number_match = re.search(r'\d+', entities.get("original_text", ""))
        step_number = int(step_number_match.group(0)) if step_number_match else 1
        
        if not self.state.get("last_solution_context"):
            return self._finalize_response({"text_response": "Mình chưa có lời giải nào trong bộ nhớ để giải thích."})

        log_chunk = self._extract_log_chunk_for_step(step_number)
        if not log_chunk:
            return self._finalize_response({"text_response": f"Mình không tìm thấy thông tin chi tiết cho bước {step_number}. Có thể phương pháp giải vừa rồi không có bước lặp."})
            
        explanation = await self.gpt_nlp.explain_simplex_step(log_chunk)
        return self._finalize_response({ "text_response": explanation or "Xin lỗi, mình chưa thể giải thích bước này.", "allow_html": True, "suggestions": [f"Giải thích bước {step_number + 1}"] })

    async def _handle_intent_request_specific_solver(self, entities: Dict):
        if not self._is_problem_defined():
            return self._finalize_response({"text_response": "Mình chưa có bài toán nào để giải lại."})
        
        # ✨ Cải tiến: Truyền định nghĩa bài toán vào để chọn solver thông minh
        solver_to_use = self._map_solver_name(entities.get("original_text", ""), self.state["current_problem_definition"])
        
        self.state["pending_action"] = {"action": "solve", "solver": solver_to_use}
        self.state["expectation"] = "awaiting_confirmation"
        confirmation_text = f"Được chứ! Mình sẽ giải lại bài toán bằng phương pháp <b>{solver_to_use}</b> nhé?"
        return self._finalize_response({"text_response": confirmation_text, "allow_html": True, "suggestions": ["Đúng vậy, giải đi", "Thôi, để sau"]})

    async def _solve_current_problem(self, solver_name: Optional[str] = None, preamble: Optional[str] = None):
        internal_def = self.state["current_problem_definition"]
        solver_format = self._convert_internal_to_solver_format(internal_def)
        if not solver_format: return self._finalize_response({"text_response": "Rất tiếc, có lỗi khi chuẩn bị dữ liệu."})

        if not solver_name:
             # ✨ Cải tiến: Logic chọn solver mặc định thông minh
            solver_name = self._map_solver_name("đơn hình", internal_def) # Giả định người dùng muốn xem bước giải
            if len(solver_format.get("variables_names_for_title_only", [])) == 2:
                solver_name = "geometric" # Ưu tiên hình học cho bài toán 2 biến
        
        self._log(f"Bắt đầu giải bằng solver mặc định: '{solver_name}'...")
        solution, logs = dispatch_solver(solver_format, solver_name=solver_name)
        
        self.state["last_solution_context"] = {"problem_definition": internal_def, "solution": solution, "logs": logs}
        
        response_parts = [f"<p>{preamble}</p>"] if preamble else []
        response_parts.append(self._format_problem_summary(internal_def))
        response_parts.append(self._format_solution_response(solution, solver_name))
        
        suggestions = ["Bắt đầu bài toán mới"]
        if solution and solution.get("status") == "Optimal":
            if solver_name not in ["geometric", "pulp_cbc"]: suggestions.insert(0, "Hiển thị cách giải")
            num_vars = len(solver_format.get("variables_names_for_title_only", []))
            if num_vars == 2 and solver_name != "geometric": suggestions.insert(0, "Giải bằng hình học")
            if "simple" in solver_name: suggestions.append("Giải lại bằng Bland")

        return self._finalize_response({ "text_response": "".join(response_parts), "allow_html": True, "suggestions": suggestions })

    async def handle_message(self, user_message: str) -> Dict[str, Any]:
        self._log(f"Đang xử lý: '{user_message}' (Chờ: {self.state['expectation']})")
        
        # ... Các luồng xử lý khác giữ nguyên ...
        if self.state["expectation"] == "awaiting_confirmation":
            self.state["expectation"] = None
            pending = self.state.pop("pending_action", None)
            if pending and ("không" not in user_message.lower() and "thôi" not in user_message.lower()):
                if pending['action'] == 'solve': return await self._solve_current_problem(pending['solver'])
            else: return self._finalize_response({"text_response": "Được rồi, nếu bạn cần gì khác cứ nói nhé!", "suggestions": ["Bắt đầu bài toán mới"]})
        
        lower_msg = user_message.lower()
        if "bài toán mẫu" in lower_msg:
             # Logic để liệt kê bài toán mẫu
             return self._handle_list_sample_problems()
        
        if self.state["expectation"] == "awaiting_sample_choice":
            return await self._handle_sample_choice(user_message)

        if any(kw in lower_msg for kw in ["bắt đầu lại", "reset", "làm mới", "bài toán mới"]):
             self.reset_state()
             return self._finalize_response({"text_response": "Đã làm mới! Mình có thể giúp gì cho bạn?", "suggestions": ['Giải bài toán mẫu']})

        parsed_lp, _ = self.lp_formula_parser(user_message)
        if parsed_lp and self._is_problem_defined(parsed_lp):
            self.state["current_problem_definition"] = parsed_lp
            return await self._solve_current_problem()

        nlp_result = self.rule_based_nlp.parse_intent_and_entities(user_message)
        intent = nlp_result.get("intent", "unknown")
        if self._is_problem_defined():
            if intent == "request_step_explanation": return await self._handle_intent_request_step_explanation(nlp_result.get("entities", {}))
            if intent == "request_specific_solver": return await self._handle_intent_request_specific_solver(nlp_result.get("entities", {}))

        response_text = await self.gpt_nlp.handle_general_conversation(user_message, self.state.get("history", [])) or "Xin lỗi, mình chưa hiểu ý bạn."
        return self._finalize_response({"text_response": response_text})
    
    # ... Các hàm định dạng và handler khác giữ nguyên ...
    def _handle_list_sample_problems(self):
        if not self.sample_problems: return self._finalize_response({"text_response": "Xin lỗi, mình chưa có sẵn bài toán mẫu nào cả."})
        response_lines = ["Mình có một vài bài toán mẫu đây, bạn muốn thử bài nào?"]
        suggestions = [f"Chọn bài toán {i+1}" for i in range(len(self.sample_problems))]
        for i, problem in enumerate(self.sample_problems): response_lines.append(f"<b>{i+1}. {problem['name']}</b>: {problem['story']}")
        self.state["expectation"] = "awaiting_sample_choice"
        return self._finalize_response({ "text_response": "<br>".join(response_lines), "allow_html": True, "suggestions": suggestions })

    async def _handle_sample_choice(self, user_message: str):
        try:
            choice_index = int(re.search(r'\d+', user_message).group(0)) - 1
            if 0 <= choice_index < len(self.sample_problems):
                chosen_problem = self.sample_problems[choice_index]
                self._log(f"Người dùng đã chọn bài toán mẫu: {chosen_problem['name']}")
                parsed_lp, _ = self.lp_formula_parser(chosen_problem['full_problem_string'])
                self.state['current_problem_definition'] = parsed_lp
                self.state['expectation'] = None
                return await self._solve_current_problem(preamble=f"<b>Đã chọn '{chosen_problem['name']}'.</b><br>" + chosen_problem['story'])
            else: raise IndexError
        except (AttributeError, ValueError, IndexError):
            return self._finalize_response({"text_response": "Lựa chọn không hợp lệ. Bạn vui lòng chọn lại từ danh sách nhé."})

    def _format_problem_summary(self, internal_def: Dict) -> str:
        obj_type = internal_def.get('objective_type', 'N/A').capitalize()
        obj_expr = self._format_coeffs_map_for_display(internal_def.get('objective_coeffs_map', {}))
        summary_html = f"<b>Bài toán của bạn:</b><ul><li><b>Mục tiêu:</b> {obj_type} Z = {obj_expr}</li>"
        if internal_def.get("constraints"):
            summary_html += "<li><b>Các ràng buộc:</b><ul style='padding-left: 20px;'>"
            for c in internal_def["constraints"]:
                lhs = self._format_coeffs_map_for_display(c.get('coeffs_map', {}))
                op = c.get('operator', '=').replace('==', '=')
                rhs = c.get('rhs', 0)
                summary_html += f"<li>{lhs} {op} {rhs:g}</li>"
            summary_html += "</ul></li>"
        summary_html += "</ul><hr style='margin: 12px 0; border-color: #e2e8f0;'>"
        return summary_html
        
    def _format_solution_response(self, solution: dict, solver_name: str) -> str:
        if not solution: return f"<p><b>Kết quả từ phương pháp '{solver_name}':</b></p><p>Rất tiếc, đã có lỗi xảy ra.</p>"
        status = solution.get("status", "Unknown")
        header = f"<p><b>Kết quả từ phương pháp '{solver_name}':</b></p>"
        body = ""
        if status == "Optimal":
            obj_val = solution.get("objective_value", 0)
            vars_map = solution.get("variables", {})
            vars_str = ", ".join([f"<b>{k}</b> = {v:.3f}" for k, v in vars_map.items() if not k.startswith('_')])
            body = f"🎉 <b>Lời giải tối ưu!</b><br>• <b>Giá trị mục tiêu:</b> {obj_val:.3f}<br>• <b>Giá trị các biến:</b> {vars_str}"
        elif status == "Infeasible": body = "Trạng thái: <b>Vô nghiệm</b>."
        elif status == "Unbounded": body = "Trạng thái: <b>Không bị chặn</b>."
        else: body = f"Trạng thái: <b>{status}</b>."
        image_html = ""
        if solution.get("plot_image_base64"):
            img_src = solution["plot_image_base64"]
            image_html = f"<br><br><div style='text-align: center;'><img src='{img_src}' alt='Biểu đồ giải bằng hình học' style='max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);' /></div>"
        return header + body + image_html

    def _format_coeffs_map_for_display(self, coeffs_map: Dict[str, float]) -> str:
        if not coeffs_map: return "0"
        terms = []
        sorted_items = sorted(coeffs_map.items())
        is_first_term = True
        for var, coeff in sorted_items:
            if abs(coeff) < 1e-9: continue
            sign = "" if is_first_term else (" + " if coeff > 0 else " - ")
            if is_first_term and coeff < 0: sign = "-"
            abs_coeff = abs(coeff)
            coeff_part = f"{abs_coeff:g}" if abs(abs_coeff - 1.0) > 1e-9 or var == '' else ""
            terms.append(f"{sign}{coeff_part}{var}")
            is_first_term = False
        return "".join(terms).strip()

    def _finalize_response(self, response: Dict) -> Dict:
        response.setdefault("text_response", "Mình chưa rõ ý bạn.")
        response.setdefault("suggestions", ["Giải bài toán mẫu", "Bắt đầu bài toán mới"])
        response.setdefault("allow_html", False)
        self.state["history"].append({"role": "assistant", "content": response["text_response"]})
        return response
