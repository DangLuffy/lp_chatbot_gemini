| Tính năng                       | Có thể dùng NlpGptParser |
| ------------------------------- | ------------------------ |
| Parse bài toán từ câu tự nhiên  | ✅                        |
| Parse từ các câu "nửa tự nhiên" | ✅                        |
| Giải thích khái niệm LP         | ✅                        |
| Giải thích trạng thái lời giải  | ✅                        |
| Giải thích từng bước giải       | ✅                        |
| Đề xuất solver phù hợp          | ✅ (có thể dùng prompt)   |
| Chat "giáo viên dạy LP"         | ✅                        |


Chắc chắn rồi! Dưới đây là giải thích về chức năng của từng parser chính trong thư mục `app/chatbot/nlp/` của bạn:

1.  **`app/chatbot/nlp/lp_parser.py`** (Cụ thể là hàm `parse_lp_problem_from_string` và `parse_expression_to_coeffs_map`)

    * **Chức năng chính**:
        * `parse_expression_to_coeffs_map`: Hàm này là một tiện ích cốt lõi. Nó nhận một chuỗi biểu thức toán học (ví dụ: `"3x1 - 2.5x2 + var_ABC"`) và phân tích nó thành một "map hệ số" (dictionary), nơi key là tên biến và value là hệ số của biến đó. Nó cũng trả về một danh sách các biến được tìm thấy theo thứ tự. Ví dụ, `"3x1 - 2.5x2 + var_ABC"` sẽ trở thành `({'x1': 3.0, 'x2': -2.5, 'var_ABC': 1.0}, ['x1', 'x2', 'var_ABC'])` [cite: uploaded:lp_chatbot_gemini/app/chatbot/nlp/lp_parser.py].
        * `parse_lp_problem_from_string`: Hàm này được thiết kế để phân tích cú pháp **toàn bộ một bài toán Quy hoạch Tuyến tính** được viết dưới dạng một chuỗi văn bản duy nhất (ví dụ: `"Maximize Z = 3x1 + 2x2 subject to x1 + x2 <= 10; 2x1 - x2 >= 0"`).
            * Nó sử dụng các biểu thức chính quy (regex) từ `rule_templates.py` để tách riêng phần hàm mục tiêu (loại tối đa hóa/tối thiểu hóa, biểu thức mục tiêu) và phần các ràng buộc [cite: uploaded:lp_chatbot_gemini/app/chatbot/nlp/lp_parser.py].
            * Sau đó, nó dùng `parse_expression_to_coeffs_map` để xử lý biểu thức mục tiêu và vế trái của từng ràng buộc.
    * **Đầu ra (của `parse_lp_problem_from_string`)**:
        * Một dictionary Python biểu diễn bài toán LP đã được phân tích. Cấu trúc này sử dụng `coeffs_map` cho cả hàm mục tiêu và các ràng buộc. Ví dụ:
            ```python
            {
                "objective_type": "maximize", # hoặc "minimize"
                "objective_coeffs_map": {"x1": 3.0, "x2": 2.0},
                "objective_variables_ordered": ["x1", "x2"], # Danh sách biến đã sắp xếp
                "constraints": [
                    {
                        "name": "c1", # Tên ràng buộc
                        "coeffs_map": {"x1": 1.0, "x2": 1.0}, # Map hệ số vế trái
                        "operator": "<=",
                        "rhs": 10.0
                    },
                    # ... các ràng buộc khác
                ]
            }
            ```
    * **Ai sử dụng**: `DialogManager` gọi trực tiếp hàm `parse_lp_problem_from_string` như một lựa chọn hàng đầu để xử lý tin nhắn của người dùng, phòng trường hợp người dùng nhập toàn bộ bài toán một lúc [cite: uploaded:lp_chatbot_gemini/app/chatbot/dialog_manager.py].

2.  **`app/chatbot/nlp/nlp_parser.py`** (Class `NlpParser`)

    * **Chức năng chính**:
        * **Nhận dạng ý định (Intent Recognition)**: Xác định mục đích chính trong câu nói của người dùng (ví dụ: người dùng muốn "định nghĩa hàm mục tiêu", "thêm một ràng buộc", "yêu cầu giải bài toán", "hỏi về một khái niệm lý thuyết", v.v.). Nó sử dụng các mẫu regex trong `rule_templates.INTENT_PATTERNS` để làm điều này [cite: uploaded:lp_chatbot_gemini/app/chatbot/nlp/nlp_parser.py].
        * **Trích xuất thực thể (Entity Extraction)**: Tìm và lấy ra các thông tin cụ thể, quan trọng từ câu nói của người dùng, ví dụ như tên solver được yêu cầu, tên khái niệm LP muốn hỏi [cite: uploaded:lp_chatbot_gemini/app/chatbot/nlp/nlp_parser.py].
        * **Phân tích từng phần của bài toán LP**: Cung cấp các hàm để xử lý các phần riêng lẻ của bài toán nếu người dùng nhập từng chút một.
            * `parse_objective_from_text`: Phân tích một chuỗi văn bản chỉ chứa hàm mục tiêu (ví dụ: `"Tối đa hóa Z = 3x1 - 2.5x2 + x3"`).
            * `parse_constraints_from_text`: Phân tích một chuỗi văn bản chứa một hoặc nhiều ràng buộc (ví dụ: `"x1 + x2 <= 10; RB2: 2x1 - x2 >= 0"`).
        * Tất cả các hàm phân tích biểu thức trong đây đều dựa trên `parse_expression_to_coeffs_map` từ `lp_parser.py` để chuyển đổi các phần toán học thành `coeffs_map`.
    * **Đầu ra**:
        * `parse_intent_and_entities`: Trả về một dictionary chứa `intent` (chuỗi) và `entities` (dictionary các thực thể).
        * `parse_objective_from_text`: Trả về một dictionary có cấu trúc tương tự phần mục tiêu của `parse_lp_problem_from_string` (bao gồm `objective_type`, `coeffs_map`, `variables_ordered`).
        * `parse_constraints_from_text`: Trả về một danh sách các dictionary, mỗi dictionary biểu diễn một ràng buộc (bao gồm `name`, `coeffs_map`, `operator`, `rhs`).
    * **Ai sử dụng**: `DialogManager` sử dụng một instance của `NlpParser` để hiểu ý định của người dùng khi họ không nhập toàn bộ bài toán một lúc, hoặc khi họ đưa ra các yêu cầu khác (như hỏi, yêu cầu giải thích, reset) [cite: uploaded:lp_chatbot_gemini/app/chatbot/dialog_manager.py].

3.  **`app/chatbot/nlp/nlp_gpt_parser.py`** (Class `NlpGptParser`)

    * **Chức năng chính**:
        * Sử dụng mô hình ngôn ngữ lớn (LLM) như Gemini của Google để thực hiện các tác vụ NLP nâng cao hơn.
        * `parse_user_request_to_lp_structure`: Hàm này nhận một tin nhắn bất kỳ từ người dùng và cố gắng sử dụng LLM để phân tích nó thành một cấu trúc bài toán LP. Prompt cho LLM (trong `gpt_prompts.py`) yêu cầu LLM trả về kết quả dưới dạng JSON với các trường như `objective_type`, `objective_expression` (dưới dạng chuỗi), `constraints` (danh sách các chuỗi ràng buộc), `user_intent`, và `clarification_needed` [cite: uploaded:lp_chatbot_gemini/app/chatbot/nlp/nlp_gpt_parser.py, uploaded:lp_chatbot_gemini/app/chatbot/nlp/gpt_prompts.py].
        * Cung cấp các hàm khác để tương tác với LLM nhằm giải thích khái niệm LP (`explain_lp_concept`), giải thích trạng thái lời giải (`explain_lp_status`), giải thích bước giải Simplex (`explain_simplex_step_details`), hoặc đề xuất solver (`recommend_solver_via_llm`) [cite: uploaded:lp_chatbot_gemini/app/chatbot/nlp/nlp_gpt_parser.py].
    * **Đầu ra (của `parse_user_request_to_lp_structure`)**:
        * Một dictionary Python được phân tích từ JSON do LLM trả về. Cấu trúc dự kiến (theo prompt) là:
            ```json
            {
              "objective_type": "maximize" | "minimize" | null,
              "objective_expression": "3x1 + 2x2", // Đây là CHUỖI
              "constraints": ["x1 + x2 <= 10", "2x1 - x2 >= 0"], // Đây là DANH SÁCH CÁC CHUỖI
              "user_intent": "define_problem" | ...,
              "clarification_needed": "Câu hỏi làm rõ nếu có" | null
            }
            ```
    * **Ai sử dụng**: `DialogManager` sử dụng một instance của `NlpGptParser` như một phương án dự phòng hoặc khi cần các khả năng xử lý ngôn ngữ tự nhiên phức tạp mà các parser dựa trên quy tắc không đáp ứng được (ví dụ, khi intent là "unknown" sau khi `NlpParser` xử lý, hoặc khi cần giải thích các khái niệm không có trong `knowledge_base.json`) [cite: uploaded:lp_chatbot_gemini/app/chatbot/dialog_manager.py]. `DialogManager` sau đó sẽ phải lấy các chuỗi `objective_expression` và `constraints` này và phân tích chúng thêm (có thể dùng lại `parse_expression_to_coeffs_map` và logic tương tự) để đưa về định dạng `coeffs_map` nội bộ.

Tóm lại, các parser này làm việc cùng nhau trong `DialogManager`:
* `lp_parser.py` xử lý trường hợp người dùng nhập toàn bộ bài toán.
* `nlp_parser.py` xử lý các tương tác từng phần và các ý định chung.
* `nlp_gpt_parser.py` là công cụ mạnh hơn để xử lý các yêu cầu phức tạp hoặc khi các phương pháp khác không thành công, đồng thời hỗ trợ các tác vụ tạo sinh văn bản (giải thích, đề xuất).

`DialogManager` sẽ tổng hợp thông tin từ các parser này, chuẩn hóa nó (thường về dạng `coeffs_map` trong `current_problem_definition`), và sau đó, khi người dùng yêu cầu giải, nó sẽ gọi `_convert_current_definition_to_solver_format` để tạo ra định dạng cuối cùng (với `coeffs` và `lhs` là list) để truyền cho các bộ giải.