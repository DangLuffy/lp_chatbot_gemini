User nhập bài toán (qua Web UI hoặc API hoặc Telegram)
    ↓
app/chatbot/nlp.py → Parse bài toán (Max/Min Z, constraints)
    ↓
app/solver/dispatcher.py → Chọn solver:
    → pulp_cbc_solver.py → giải bài lớn
    → simplex_manual_solver.py → giải có log từng bước
    ↓
Trả kết quả + log (tuỳ solver)
    ↓
app/chatbot/dialog_manager.py → xử lý phản hồi bot
    ↓
app/api/routes.py → trả response về Web UI hoặc API
    ↓
templates/index.html hoặc Chat API → Hiển thị kết quả cho user


/linear-programming-chatbot
|
|-- main.py                      # Entry point tổng của ứng dụng
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- /app                         # App chính (Web API, Chatbot)
|   |
|   |-- /api                     # API REST cho Web hoặc Client
|   |   |-- __init__.py
|   |   |-- routes.py            # Định tuyến API
|   |   `-- handlers.py          # Logic xử lý API (nhận bài toán, trả kết quả, trả log từng bước)
|   |
|   |-- /chatbot                 # Bot NLP (có thể dùng cho Web chat, Telegram, ... )
|   |   |-- __init__.py
|   |   |-- nlp.py               # NLP xử lý text: parse LP từ text
|   |   |-- dialog_manager.py    # Quản lý hội thoại, trạng thái hội thoại
|   |   `-- templates            # Giao diện Web nếu chatbot dạng web
|   |       `-- index.html
|   |
|   |-- /solver                  # Bộ giải QHTT (giải thực tế)
|   |   |-- __init__.py
|   |   |-- pulp_cbc_solver.py   # Dùng Pulp + CBC
|   |   |-- simplex_manual_solver.py # Dùng Simplex manual có log từng bước
|   |   |-- utils.py             # Các hàm tiện ích chung (convert input → matrix)
|   |   `-- dispatcher.py        # Gọi solver theo tên
|   |
|   `-- __init__.py
|
|-- /core                        # Cấu hình và logging
|   |-- __init__.py
|   |-- config.py                # Config (API keys, env, model path)
|   `-- logger.py                # Cấu hình logging
|
|-- /data                        # Dữ liệu mẫu + training NLP
|   |-- /examples
|   |   |-- problem1.json
|   |   |-- problem2.txt
|   |
|   `-- /training_data
|       |-- intents.json
|       |-- entities.json
|
|-- /static                      # Frontend assets (nếu dùng Web UI)
|   |-- /css
|   |   `-- style.css
|   |-- /js
|   |   `-- main.js
|
|-- /tests                       # Unit test
|   |-- __init__.py
|   |-- test_solver.py
|   `-- test_api.py
