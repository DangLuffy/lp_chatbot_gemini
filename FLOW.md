┌──────────────────────────────┐
│ 1. Người dùng nhập bài toán LP│
│    ← textarea Web / JSON API  │
│    ← tin nhắn Telegram        │
└───────────────┬───────────────┘
                │ (chuỗi văn bản **hoặc** dict JSON input)
                ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. chatbot/nlp.py                                               │
│    • Nếu input là chuỗi text:                                   │
│        – Tách dòng “Max/Min Z = …”                              │
│        – Trích hệ số, tên biến, ràng buộc (<=, >=, =)           │
│        – Chuẩn hoá dấu cách, ký hiệu (≤, >=, =)                 │
│        – Trả dict:                                              │
│          {                                                      │
│            "objective": "min" / "max",                          │
│            "coeffs": [c1, …, cn],                               │
│            "variables_names_for_title_only": ["x1", "x2",...],  │
│            "constraints": [                                     │
│               {"name": "R1", "lhs": [...], "op": "<=", "rhs": b} │
│               ...                                               │
│            ]                                                    │
│          }                                                      │
│    • Nếu input đã là dict JSON chuẩn (giống problem_f72cfa_input)│
│        – Bỏ qua bước NLP → dùng luôn dict.                      │
└───────────────┬────────────────────────────────────────────────┘
                │ (dict chuẩn của bài toán LP)
                ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. solver/dispatcher.py                                        │
│    • Nhìn tham số solver_name:                                 │
│      - "pulp_cbc"        → gọi pulp_cbc_solver                  │
│      - "simple_dictionary" → gọi simple_dictionary_solver      │
│      - "simplex_bland"   → gọi simplex_bland_solver            │
│      - "auxiliary"       → gọi auxiliary_problem_solver        │
│      - "two_phase"       → gọi two_phase_simplex_solver        │
│      - "dual_simplex"    → gọi dual_simplex_solver             │
│    • Hàm trả ra dict kết quả:                                  │
│      {                                                         │
│        "status": "Optimal" / "Infeasible" / ...,                │
│        "solution": [x1, …, xn],                                │
│        "objective_value": Z,                                   │
│        "step_log": [... hoặc None]                             │
│      }                                                         │
└───────────────┬────────────────────────────────────────────────┘
                │ (kết quả + log)
                ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. chatbot/dialog_manager.py                                   │
│    • Ghép câu trả lời thân thiện:                              │
│      "Nghiệm tối ưu: x1 = …, x2 = …, Z = …"                     │
│    • Nếu có step_log:                                          │
│      – Thêm nút "Xem chi tiết từng bước"                       │
│    • Lưu session context (cho câu hỏi tiếp theo "vì sao?", "giải lại") │
└───────────────┬────────────────────────────────────────────────┘
                │ (payload format sẵn)
                ▼
┌────────────────────────────────────────────────────────────────┐
│ 5. api/routes.py                                               │
│    • Nếu Web: render template Jinja:                           │
│      → return render_template("index.html", **payload)         │
│    • Nếu REST API:                                             │
│      → return JSONResponse(payload)                           │
│    • Nếu Telegram:                                             │
│      → bot.send_message(chat_id, text, …)                      │
└───────────────┬────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────┐
│ 6. Giao diện cuối                                              │
│    • Web (index.html) hiển thị:                                │
│      – Kết quả tóm tắt                                         │
│      – (Tùy chọn) bảng log từng bước Simplex / Hai Pha / Đối ngẫu│
│    • Telegram hiển thị tương tự                                │
└────────────────────────────────────────────────────────────────┘


/linear-programming-chatbot
|
|-- main.py                     # Entry point tổng của ứng dụng
|-- requirements.txt            # Các thư viện Python cần thiết
|-- README.md                   # Mô tả dự án và hướng dẫn
|-- .gitignore                  # Các tệp và thư mục Git sẽ bỏ qua
|-- FLOW.md                     # Sơ đồ luồng hoạt động của ứng dụng (đã cập nhật)
|
|-- /app                        # App chính (Web API, Chatbot)
|   |
|   |-- /api                    # API REST cho Web hoặc Client
|   |   |-- __init__.py
|   |   |-- routes.py           # Định tuyến API (gom nhóm các handlers)
|   |   `-- handlers.py         # Logic xử lý API (nhận bài toán, gọi dispatcher, trả kết quả)
|   |
|   |-- /chatbot                # Bot NLP (Web chat, Telegram, ...)
|   |   |-- __init__.py
|   |   |-- nlp.py              # NLP xử lý text: parse LP từ text sang định dạng NLP mới
|   |   |-- dialog_manager.py   # Quản lý hội thoại, gọi NLP và dispatcher
|   |   `-- /templates          # Giao diện Web nếu chatbot dạng web
|   |       `-- index.html
|   |
|   |-- /solver                 # Bộ giải QHTT (giải thực tế)
|   |   |-- __init__.py
|   |   |-- pulp_cbc_solver.py  # Dùng thư viện PuLP + CBC (không đổi)
|   |   |-- geometric_solver.py # Giải bằng phương pháp hình học cho 2 biến (không đổi)
|   |   |-- utils.py            # Các hàm tiện ích chung:
|   |   |   |                       # - normalize_problem_data_from_nlp() (chuyển định dạng NLP mới sang cũ)
|   |   |   |                       # - convert_problem_to_matrix_form()
|   |   |
|   |   |-- base_simplex_dictionary_solver.py # LỚP CHA cho các solver Đơn hình từ điển
|   |   |
|   |   |-- simple_dictionary_solver.py # KẾ THỪA: Đơn hình từ điển, quy tắc Dantzig (thay thế simplex_manual_solver.py cũ)
|   |   |-- simplex_bland_solver.py     # KẾ THỪA: Đơn hình từ điển, quy tắc Bland
|   |   |-- auxiliary_problem_solver.py # KẾ THỪA: Đơn hình, dùng bài toán bổ trợ với biến x0 (theo hình ảnh)
|   |   |-- two_phase_simplex_solver.py # KẾ THỪA: Phương pháp Hai Pha chuẩn (dùng biến nhân tạo A_i)
|   |   |-- dual_simplex_solver.py      # KẾ THỪA: Thuật toán Đơn hình Đối ngẫu
|   |   |
|   |   `-- dispatcher.py       # Gọi solver theo tên (cần cập nhật để bao gồm các solver mới)
|   |
|   `-- __init__.py
|
|-- /core                       # Cấu hình và logging
|   |-- __init__.py
|   |-- config.py               # Config (API keys, env, model path)
|   `-- logger.py               # Cấu hình logging
|
|-- /data                       # Dữ liệu mẫu + training NLP (chưa có nội dung)
|   |-- /examples
|   |   |-- problem1.json
|   |   |-- problem2.txt
|   |
|   `-- /training_data
|       |-- intents.json
|       |-- entities.json
|
|-- /static                     # Frontend assets (nếu dùng Web UI)
|   |-- /css
|   |   `-- style.css
|   |-- /js
|   |   `-- main.js
|
|-- /tests                      # Unit test
|   |-- __init__.py
|   |-- test_solver.py          # Cần cập nhật để test các solver mới
|   |-- test_api.py
|   `-- test_chatbot_nlp.py     # (Hoặc một tệp test riêng cho NLP)



