# /app/chatbot/dialog_manager.py

import logging
from typing import Dict, Any, List

# Import các thành phần cần thiết
from .nlp import extract_lp_from_text
from app.solver.dispatcher import dispatch_solver

logger = logging.getLogger(__name__)

# Quản lý trạng thái hội thoại cho mỗi người dùng (nếu cần)
# Key: user_id, Value: conversation_state
# Trong ví dụ đơn giản này, chúng ta không lưu trạng thái phức tạp.
conversation_states: Dict[str, Any] = {}

class DialogManager:
    """
    Quản lý logic hội thoại của chatbot.
    """
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        # Khởi tạo trạng thái nếu người dùng chưa có
        if self.user_id not in conversation_states:
            conversation_states[self.user_id] = {"history": []}
        
        self.state = conversation_states[self.user_id]
        self.logs = []

    def _log(self, message: str):
        self.logs.append(message)
        logger.info(message)

    def handle_message(self, user_message: str) -> str:
        """
        Xử lý tin nhắn từ người dùng và trả về phản hồi của bot.
        """
        self._log(f"Handling message from user '{self.user_id}': '{user_message}'")
        self.state["history"].append({"user": user_message})

        # Bước 1: Cố gắng phân tích toàn bộ tin nhắn như một bài toán LP
        problem_data, nlp_logs = extract_lp_from_text(user_message)
        self.logs.extend(nlp_logs)

        if not problem_data:
            # Nếu phân tích thất bại, trả về thông báo hướng dẫn
            self._log("NLP parsing failed. Replying with a help message.")
            response = "Tôi không hiểu yêu cầu của bạn. Vui lòng nhập bài toán theo định dạng, ví dụ:\n" \
                       "'Maximize 3x1 + 2x2 subject to x1+x2<=10, 2x1+x2<=15'"
            self.state["history"].append({"bot": response})
            return response

        # Bước 2: Nếu phân tích thành công, gọi bộ giải
        self._log(f"NLP parsing successful. Dispatching to solver.")
        # Mặc định sử dụng pulp_cbc, có thể cho phép người dùng chọn sau này
        solution, solver_logs = dispatch_solver(problem_data, solver_name="pulp_cbc")
        self.logs.extend(solver_logs)
        
        # Bước 3: Định dạng kết quả để trả lời người dùng
        response = self._format_solution_response(solution, problem_data)
        self._log(f"Formatted response: {response}")
        self.state["history"].append({"bot": response})
        
        return response

    def _format_solution_response(self, solution: Dict[str, Any], problem: Dict[str, Any]) -> str:
        """
        Chuyển đổi kết quả từ bộ giải thành một chuỗi văn bản thân thiện.
        """
        if not solution:
            return "Rất tiếc, đã xảy ra lỗi trong quá trình giải bài toán."

        status = solution.get("status", "Unknown")
        
        if status == "Optimal":
            obj_value = solution.get("objective_value", 0)
            variables = solution.get("variables", {})
            
            response_parts = [
                f"🎉 Lời giải tối ưu đã được tìm thấy!",
                f"Giá trị hàm mục tiêu ({problem['objective']['type']}): {obj_value:.4g}",
                "Giá trị của các biến:"
            ]
            for var, val in variables.items():
                response_parts.append(f"  - {var} = {val:.4g}")
            
            return "\n".join(response_parts)
        
        elif status == "Infeasible":
            return "Bài toán không có lời giải khả thi (các ràng buộc mâu thuẫn với nhau)."
            
        elif status == "Unbounded":
            return "Bài toán không bị chặn (giá trị hàm mục tiêu có thể tiến tới vô cùng)."
        
        else:
            return f"Không thể tìm thấy lời giải tối ưu. Trạng thái của bộ giải là: {status}."

    def get_logs(self) -> List[str]:
        """Trả về toàn bộ logs của quá trình xử lý."""
        return self.logs
