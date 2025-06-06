# Luồng Hoạt Động và Cấu Trúc Dự Án LP Chatbot

## I. Luồng Hoạt Động Chính của Ứng Dụng

Sơ đồ này mô tả các bước chính khi người dùng tương tác với chatbot để giải một bài toán Quy hoạch Tuyến tính (LP).

```mermaid
graph TD
    A[1. Người dùng nhập bài toán LP <br/> ← Giao diện Web / JSON API] --> B{2. Phân tích Yêu cầu <br/> (DialogManager + NLP Parsers)};

    subgraph NLP & Chuẩn Bị Dữ Liệu
        B --> B1[app/chatbot/nlp/lp_parser.py <br/> (Parse toàn bộ bài toán LP dạng chuỗi)];
        B --> B2[app/chatbot/nlp/nlp_parser.py <br/> (Parse ý định, thực thể, từng phần bài toán)];
        B --> B3[app/chatbot/nlp/nlp_gpt_parser.py <br/> (Parse bằng LLM nếu cần)];
        B1 --> B_DM{DialogManager <br/> Tổng hợp về dạng trung gian <br/> (dùng coeffs_map)};
        B2 --> B_DM;
        B3 --> B_DM_GPT_Parse[DialogManager <br/> _parse_gpt_structure_to_internal_format <br/> (Chuỗi -> coeffs_map)];
        B_DM_GPT_Parse --> B_DM;
        B_DM --> C[DialogManager <br/> _convert_current_definition_to_solver_format <br/> (coeffs_map -> "Định dạng A")];
    end

    C --> D{3. Chuẩn Hóa & Điều Phối Solver};

    subgraph Chuẩn Hóa & Điều Phối
        D -- "Định dạng A" --> D_PulpGeo[app/solver/dispatcher.py <br/> (Cho Pulp, Geometric)];
        D_PulpGeo --> D_Pulp[pulp_cbc_solver.py];
        D_PulpGeo --> D_Geo[geometric_solver.py];

        D -- "Định dạng A" --> D_Std_Wrap[Hàm Bao Bọc Solver Simplex <br/> (ví dụ: solve_with_simple_dictionary)];
        D_Std_Wrap --> D_Std[app/solver/utils.py <br/> standardize_problem_for_simplex <br/> ("Định dạng A" -> "Định dạng A Chuẩn Hóa": <br/> mục tiêu "min", ràng buộc "<=")];
        D_Std -- "Định dạng A Chuẩn Hóa" --> D_Dispatch_Simplex[app/solver/dispatcher.py <br/> (Cho các solver Simplex)];
    end

    subgraph Solvers Simplex
        D_Dispatch_Simplex --> E1[simple_dictionary_solver.py];
        D_Dispatch_Simplex --> E2[simplex_bland_solver.py];
        D_Dispatch_Simplex --> E3[auxiliary_problem_solver.py <br/> (Xử lý Pha 1 với x0_aux)];
        %% D_Dispatch_Simplex --> E4[two_phase_simplex_solver.py <br/> (Nếu có, dùng biến nhân tạo Ai)];
        %% D_Dispatch_Simplex --> E5[dual_simplex_solver.py];
    end

    E1 --> F[4. Trả Kết Quả Giải];
    E2 --> F;
    E3 --> F;
    %% E4 --> F;
    %% E5 --> F;
    D_Pulp --> F;
    D_Geo --> F;

    F --> G[5. DialogManager <br/> Định dạng câu trả lời, <br/> quản lý hội thoại];
    G --> H[6. Hiển Thị Kết Quả <br/> → Giao diện Web / Phản hồi API];

style A fill:#f9f,stroke:#333,stroke-width:2px
style B fill:#ccf,stroke:#333,stroke-width:2px
style C fill:#lightgreen,stroke:#333,stroke-width:2px
style D fill:#lightblue,stroke:#333,stroke-width:2px
style F fill:#orange,stroke:#333,stroke-width:2px
style G fill:#ccf,stroke:#333,stroke-width:2px
style H fill:#f9f,stroke:#333,stroke-width:2px
|-- main.py                     # Entry point tổng của ứng dụng
|-- requirements.txt            # Các thư viện Python cần thiết
|-- README.md                   # Mô tả dự án và hướng dẫn
|-- .gitignore                  # (Nên có) Các tệp và thư mục Git sẽ bỏ qua
|-- FLOW.md                     # Sơ đồ luồng hoạt động của ứng dụng (Tệp này)
|
|-- /app                        # App chính (Web API, Chatbot)
|   |
|   |-- /api                    # API REST cho Web hoặc Client
|   |   |-- __init__.py
|   |   |-- routes.py           # Định tuyến API
|   |   `-- handlers.py         # Logic xử lý API
|   |
|   |-- /chatbot                # Logic Chatbot và NLP
|   |   |-- __init__.py
|   |   |-- /nlp                # Package xử lý ngôn ngữ tự nhiên
|   |   |   |-- __init__.py
|   |   |   |-- lp_parser.py    # Parse chuỗi LP đầy đủ -> coeffs_map
|   |   |   |-- nlp_parser.py   # Parse intent, entity, từng phần bài toán -> coeffs_map
|   |   |   |-- nlp_gpt_parser.py # Parse bằng LLM -> chuỗi biểu thức
|   |   |   |-- rule_templates.py # Regex patterns cho nlp_parser
|   |   |   |-- gpt_prompts.py  # Prompts cho nlp_gpt_parser
|   |   |   `-- knowledge_base.json # Cơ sở tri thức
|   |   |
|   |   |-- dialog_manager.py   # Quản lý hội thoại, gọi NLP, chuẩn hóa sang "Định dạng A", gọi dispatcher
|   |   |-- web_routes.py       # Định tuyến cho giao diện web chatbot
|   |   `-- /templates
|   |       `-- index.html      # Giao diện web
|   |
|   |-- /solver                 # Các bộ giải Quy hoạch Tuyến tính
|   |   |-- __init__.py
|   |   |-- pulp_cbc_solver.py  # Dùng thư viện PuLP + CBC (nhận "Định dạng A")
|   |   |-- geometric_solver.py # Giải bằng phương pháp hình học cho 2 biến (nhận "Định dạng A")
|   |   |-- utils.py            # Hàm tiện ích: standardize_problem_for_simplex()
|   |   |
|   |   |-- base_simplex_dictionary_solver.py # LỚP CHA cho solver Đơn hình từ điển (nhận "Định dạng A Chuẩn Hóa")
|   |   |
|   |   |-- simple_dictionary_solver.py # KẾ THỪA: Đơn hình từ điển, quy tắc Dantzig
|   |   |-- simplex_bland_solver.py     # KẾ THỪA: Đơn hình từ điển, quy tắc Bland
|   |   |-- auxiliary_problem_solver.py # KẾ THỪA: Đơn hình, dùng bài toán bổ trợ với x0_aux 
|   |   |-- ## two_phase_simplex_solver.py ## (Cân nhắc nếu cần phương pháp Hai Pha chuẩn dùng biến nhân tạo Ai riêng biệt)
|   |   |-- ## dual_simplex_solver.py ## (Nếu có)
|   |   |
|   |   `-- dispatcher.py       # Gọi solver theo tên
|   |
|   `-- __init__.py
|
|-- /core                       # Cấu hình và logging chung
|   |-- __init__.py
|   |-- config.py               # Config (API keys, env, model path)
|   `-- logger.py               # Cấu hình logging
|
|-- /data                       # (Tùy chọn) Dữ liệu mẫu, huấn luyện
|   |-- /examples
|   |   `-- problem1.json
|
|-- /static                     # Frontend assets
|   |-- /css
|   |   `-- style.css
|   |-- /js
|   |   `-- main.js
|
|-- /tests                      # Unit test
|   |-- __init__.py
|   |-- test_solver.py          # Cần cập nhật để test với "Định dạng A" và "Định dạng A Chuẩn Hóa"
|   |-- test_api.py
|   `-- test_chatbot_nlp.py     # Test các parser trong app/chatbot/nlp/
