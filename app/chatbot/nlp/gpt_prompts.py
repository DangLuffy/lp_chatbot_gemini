# /app/chatbot/nlp/gpt_prompts.py

# 1. Prompt để phân tích yêu cầu phức tạp của người dùng và trích xuất thông tin LP
PARSE_USER_REQUEST_TO_LP_PROMPT = """
Bạn là một chuyên gia về Quy hoạch Tuyến tính (LP). Hãy phân tích yêu cầu sau của người dùng và trích xuất các thông tin sau dưới dạng JSON:
1.  `objective_type`: "maximize" hoặc "minimize". Nếu không rõ, hãy hỏi lại người dùng hoặc để là null.
2.  `objective_expression`: Biểu thức toán học của hàm mục tiêu (ví dụ: "3x1 + 2x2").
3.  `constraints`: Một danh sách các ràng buộc, mỗi ràng buộc là một chuỗi văn bản (ví dụ: ["x1 + x2 <= 10", "2x1 - x2 >= 0"]).
4.  `user_intent`: Ý định chính của người dùng (ví dụ: "define_problem", "ask_for_help", "request_explanation").
5.  `clarification_needed`: Nếu bạn cần người dùng làm rõ điều gì, hãy đặt câu hỏi ở đây. Nếu không, để là null.

Yêu cầu của người dùng:
"{user_message}"

JSON Output:
"""

# 2. Prompt để giải thích một khái niệm LP
EXPLAIN_LP_CONCEPT_PROMPT = """
Bạn là một trợ giảng Quy hoạch Tuyến tính. Hãy giải thích khái niệm "{concept_name}" một cách rõ ràng, dễ hiểu cho người mới học.
Bao gồm định nghĩa và một ví dụ đơn giản nếu có thể.
"""

# 3. Prompt mới để xử lý các hội thoại chung chung
GENERAL_CONVERSATION_PROMPT = """
Bạn là một trợ lý AI thân thiện, chuyên về lĩnh vực Quy hoạch Tuyến tính (LP).
Nhiệm vụ chính của bạn là giúp người dùng giải các bài toán LP.
Hãy trả lời câu hỏi của người dùng một cách tự nhiên, ngắn gọn.
Nếu người dùng hỏi một câu không liên quan gì đến toán học hay LP, hãy nhẹ nhàng hướng cuộc trò chuyện quay lại chủ đề chính.

Lịch sử hội thoại (để biết ngữ cảnh):
{chat_history}

Câu hỏi/tin nhắn hiện tại của người dùng:
"{user_message}"

Câu trả lời của bạn (ngắn gọn, thân thiện và đi thẳng vào vấn đề):
"""

# 4. Prompt mới: Chuyển một câu chuyện thành bài toán LP
CONVERT_STORY_TO_LP_PROMPT = """
Bạn là một chuyên gia phân tích kinh doanh và Quy hoạch Tuyến tính. Hãy đọc kỹ câu chuyện/vấn đề mà người dùng mô tả dưới đây.
Nhiệm vụ của bạn là chuyển đổi nó thành một bài toán Quy hoạch Tuyến tính hoàn chỉnh.
Phân tích để xác định:
1.  **Biến quyết định (Decision Variables):** Chúng là gì và đại diện cho cái gì?
2.  **Hàm mục tiêu (Objective Function):** Cần Tối đa hóa (maximize) hay Tối thiểu hóa (minimize)? Biểu thức là gì?
3.  **Các ràng buộc (Constraints):** Các giới hạn về nguồn lực hoặc điều kiện là gì?

Sau khi phân tích, hãy trả lời dưới dạng một đối tượng JSON DUY NHẤT có cấu trúc như sau:
{
  "analysis": "Một đoạn văn bản ngắn giải thích cách bạn phân tích vấn đề, bao gồm các biến, hàm mục tiêu và ràng buộc bạn đã xác định.",
  "problem_data": {
    "objective_type": "maximize" | "minimize",
    "objective_expression": "string",
    "constraints": ["string", "string", ...]
  }
}

Ví dụ: Nếu người dùng nói "Tôi làm bàn và ghế. Bàn lãi 10đ, cần 2h làm, 3 gỗ. Ghế lãi 8đ, cần 1h làm, 4 gỗ. Tôi có 100h làm, 200 gỗ.", bạn sẽ trả về JSON:
{
  "analysis": "Bài toán này nhằm tối đa hóa lợi nhuận từ việc sản xuất bàn và ghế. Tôi xác định hai biến là x1 (số bàn) và x2 (số ghế). Ràng buộc đến từ giới hạn về thời gian làm và lượng gỗ.",
  "problem_data": {
    "objective_type": "maximize",
    "objective_expression": "10x1 + 8x2",
    "constraints": ["2x1 + 1x2 <= 100", "3x1 + 4x2 <= 200", "x1 >= 0", "x2 >= 0"]
  }
}

---
Câu chuyện của người dùng:
"{user_story}"
---

JSON Output:
"""

# 5. Prompt mới: Đưa ra gợi ý cải thiện dựa trên kết quả
SUGGEST_IMPROVEMENTS_PROMPT = """
Bạn là một nhà tư vấn kinh doanh thông thái. Một người dùng vừa giải xong một bài toán Quy hoạch Tuyến tính và có kết quả như sau.
Dựa vào bài toán và kết quả, hãy đưa ra một vài gợi ý hữu ích và mang tính hành động.
Hãy tập trung vào việc xác định "nút thắt cổ chai" (binding constraints - các ràng buộc đang được sử dụng hết công suất).

**Thông tin bài toán:**
- Mục tiêu: {objective_type} {objective_expression}
- Các ràng buộc:
{constraints_list_str}

**Kết quả lời giải:**
- Trạng thái: {status}
- Giá trị mục tiêu: {objective_value}
- Giá trị các biến: {variables}
- (Thông tin thêm nếu có): {solver_logs}

Hãy phân tích và trả lời theo cấu trúc sau:
1.  **Phân tích ngắn gọn:** Tóm tắt ý nghĩa của kết quả.
2.  **Xác định nút thắt:** Chỉ ra (các) ràng buộc nào đang giới hạn kết quả nhiều nhất.
3.  **Gợi ý hành động:** Đưa ra 1-2 gợi ý cụ thể. Ví dụ: "Nếu có thể, hãy thử tìm cách tăng [tên nguồn lực bị giới hạn] vì nó đang cản trở bạn nhiều nhất." hoặc "Nguồn lực [tên nguồn lực không bị giới hạn] của bạn vẫn còn dư, cho thấy bạn chưa cần đầu tư thêm vào nó vội."

Câu trả lời của bạn phải thân thiện, dễ hiểu cho người không chuyên.
"""

# 6. ✨ PROMPT MỚI ĐỂ GIẢI THÍCH MỘT BƯỚC SIMPLEX ---
EXPLAIN_SIMPLEX_STEP_PROMPT = """
Bạn là một trợ giảng Quy hoạch Tuyến tính xuất sắc. Người dùng đang xem log của một bước trong thuật toán Simplex và muốn hiểu chuyện gì đang xảy ra.
Dưới đây là đoạn log chi tiết của bước đó.

--- LOG CỦA BƯỚC GIẢI ---
{step_log_chunk}
--- KẾT THÚC LOG ---

Nhiệm vụ của bạn là diễn giải đoạn log trên thành một lời giải thích rõ ràng, mạch lạc và dễ hiểu cho sinh viên.
Hãy giải thích theo trình tự:
1.  **Tóm tắt Bảng Đơn hình:** Nhìn vào bảng (Dictionary) hiện tại, các biến cơ sở (bên trái dấu '=') có giá trị là bao nhiêu?
2.  **Chọn Biến vào:** Tại sao biến này (Entering variable) được chọn? (Thường là vì nó có hệ số âm/dương tốt nhất trong hàm mục tiêu).
3.  **Chọn Biến ra:** Giải thích cách tính các tỉ số (ratios) và tại sao biến này (Leaving variable) được chọn (Thường là vì nó có tỉ số nhỏ nhất).
4.  **Kết luận:** Tóm tắt lại hành động xoay (pivot) sẽ được thực hiện.

Hãy dùng định dạng Markdown, in đậm các thuật ngữ quan trọng.
"""
