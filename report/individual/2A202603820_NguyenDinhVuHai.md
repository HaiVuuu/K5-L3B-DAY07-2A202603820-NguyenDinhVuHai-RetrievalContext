# Báo cáo Day 7

- **Họ và tên:** Nguyễn Đình Vũ Hải
- **MSSV:** 2A202603820
- **Repository:** `K5-L3B-DAY07-2A202603820-NguyenDinhVuHai-RetrievalContext`

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu: `data/harrypotter.pdf` (*Harry Potter and the Sorcerer's Stone*, Book 1, Chapters 1–4), sử dụng cho mục đích học tập và nghiên cứu nội bộ môn học; không redistribute toàn văn PDF lên repo công khai nếu chưa được phép.
  - Phạm vi trích xuất: **46 trang** nội dung (từ trang 12 đến trang 57 của bản PDF), được chia thành **206 chunks** theo ranh giới đoạn văn (`chunk_strategy: "paragraph"`), lưu tại `data/chunks.json`.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: **ChromaDB** persistent local tại `chroma_db/` (collection `hp_strat_b`), kế thừa trực tiếp từ Day 6.
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (chạy local qua thư viện `sentence-transformers` với `local_files_only=True`, độ đo cosine similarity, kích thước vector **384 chiều**, chi phí `$0.00`).
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent model: Google Gemini API **`gemini-3.1-flash-lite`** (`temperature=0.0`, `responseMimeType="application/json"`), cấu hình qua `.env` (`GEMINI_API_KEY`, `GEMINI_MODEL`).
  - Cách chạy: Gọi từ thư mục gốc bằng Python 3.11+ (`.venv`), có cơ chế giới hạn tốc độ `--delay 20` giây/call (~3 RPM) để tuân thủ giới hạn 15 RPM của Gemini Free Tier, `--max-requests` và `--skip-existing` để lưu/khôi phục trace an toàn.
  - Phân chia `dev`/`test` (`data/questions.json`, tổng **24 câu**, tách biệt hoàn toàn không rò rỉ):
    - **`dev` (17 câu):** 15 câu có bằng chứng (`q01–q15`) + 2 câu thiếu bằng chứng (`q21–q22`). Dùng để phát triển, so sánh phương pháp retrieval, phân tích lỗi và chốt cấu hình.
    - **`test` (7 câu):** 5 câu có bằng chứng (`q16–q20`) + 2 câu thiếu bằng chứng (`q23–q24`). Giữ độc lập hoàn toàn, chỉ chạy **1 lần duy nhất** sau khi chốt cấu hình trên `dev`.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có evidence (`answerable: true`, `q01–q20`):** Nội dung đáp án xuất hiện trực tiếp trong Chapters 1–4 của tài liệu, đi kèm trường `gold` ghi rõ `source_id`, `page` và đoạn trích `evidence`.
  - **Câu thiếu evidence (`answerable: false`, `q21–q24`):** Câu hỏi về các sự kiện thuộc các tập Harry Potter sau không tồn tại trong corpus Chapters 1–4 (Patronus/Dementors ở Book 3, Goblet of Fire ở Book 4, nhật ký Tom Riddle ở Book 2, trường sinh linh giá ở Book 7).
  - **Tiêu chí chấm (theo `BENCHMARK.md` & `starter/agent_metrics.py`):**
    - `answer_grade` (`0–2` cho câu có evidence): `2` = đúng hoàn toàn, bám sát ngữ cảnh, có citation chính xác; `1` = đúng một phần hoặc thiếu ý; `0` = sai, bịa đặt hoặc từ chối nhầm (`false_abstention`).
    - `citation_support` (`true`/`false`): Mỗi citation `[source_id:page:chunk_id]` phải tồn tại trong `context_ids` thực tế của lần chạy và đoạn văn đó phải trực tiếp chứng minh cho câu trả lời.
    - `abstention_correct` (`true`/`false`): Câu thiếu evidence phải trả về đúng chuỗi từ chối chuẩn `"Không đủ bằng chứng để trả lời."` và không kèm citation bịa.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** (`dev-dense`) | Dense retrieval (ChromaDB `all-MiniLM-L6-v2`), `top_k=3`, auto chapter filter, không rerank, `budget=800` chars | Thiết lập mốc chuẩn từ vector store Day 6; kiểm tra giới hạn của dense đơn thuần và budget nhỏ | dev | 3/15 (20.0%) | 9/15 (60.0%) | 3/15 (20.0%) | 25.93 ms/câu / $0.00 |
| **Phương pháp 2** (`dev-bm25`) | Lexical BM25 Okapi ($k_1=1.5, b=0.75$), `top_k=3`, auto chapter filter, không rerank, `budget=800` chars | Giả thuyết: Truy vấn chứa nhiều tên riêng (*Grunnings*, *Put-Outer*, *Privet Drive*) cần khớp từ khóa chính xác | dev | 9/15 (60.0%) | 12/15 (80.0%) | 9/15 (60.0%) | 1.13 ms/câu / $0.00 |
| **Phương pháp 3** (`dev-tfidf`) | Lexical TF-IDF cosine similarity, `top_k=3`, auto chapter filter, không rerank, `budget=800` chars | Giả thuyết: Trọng số nghịch đảo tần suất tài liệu (IDF) làm nổi bật danh từ riêng hiếm trong 206 chunks | dev | 11/15 (73.3%) | 12/15 (80.0%) | 11/15 (73.3%) | 1.13 ms/câu / $0.00 |
| **Cải thiện 1** (`config1`) | Agent `hybrid_search` (Weighted RRF: `1.5*TF-IDF + 1.2*BM25 + 1.0*Dense` trên top-20), `top_k=3`, auto chapter/page filter, `budget=2400` chars, `max_searches=2` | **Thay đổi 1:** Kết hợp lexical + dense qua RRF và tăng `budget` từ `800` lên `2400` chars để nhét đủ 2–3 chunks thay vì bị cắt ở chunk 1 | dev | 13/15 (86.7%)* | 13/15 (86.7%) | 13/15 (86.7%) | 46.50 s/câu** / $0.0061 |
| **Cải thiện 2** (`config2`) | Agent `hybrid_search` (Weighted RRF), `top_k=3`, auto filter, `budget=3600` chars, `max_searches=3`, bật `expand_neighbors=True` (`pos+1`, `pos-1` liền trang) | **Thay đổi 2:** Mở rộng thêm chunk liền kề (`pos+1`/`pos-1`) và tăng `budget=3600` chars để vá các đoạn hội thoại/bằng chứng bị cắt qua ranh giới chunk hoặc trang | dev | 14/15 (93.3%)* | 14/15 (93.3%) | 14/15 (93.3%) | 46.78 s/câu** / $0.0069 |
| **Cấu hình đã chọn** (`config2`) | Giữ nguyên toàn bộ tham số `config2` đã chốt trên `dev` (`hybrid_search` RRF, `top_k=3`, `budget=3600` chars, `max_searches=3`, `expand_neighbors=True`) | Đánh giá độc lập 1 lần duy nhất sau khi chốt trên `dev` | test | 3/5 (60.0%)*** | 5/5 (100% chấm tay) | 5/5 (100% chấm tay) | 42.98 s/câu** / $0.0028 |
| **Baseline** (`test-dense`) | Dense retrieval (ChromaDB `all-MiniLM-L6-v2`), `top_k=3`, auto chapter filter, không rerank, `budget=800` chars | Đối chiếu mốc chuẩn offline trên tập `test` | test | 2/5 (40.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 294.14 ms/câu / $0.00 |

*\*Ghi chú bảng: Với cấu hình Agent (`config1`, `config2`), cột Hit@1 / Evidence Hit@3 phản ánh tỷ lệ bằng chứng nằm trong tập chunk được agent đưa vào context (`context_hit_rate`).*
*\*\*Thời gian trung bình ~43–47s/câu bao gồm `40s` nghỉ chủ động (`--delay 20` × 2 LLM calls/câu) để tuân thủ giới hạn 15 RPM của Gemini Free Tier; thời gian truy xuất retrieval thực tế chỉ mất `20–170 ms/câu`.*
*\*\*\*Trên `test`, so khớp ID gold tự động đạt `3/5 (60.0%)`, nhưng kiểm chứng thủ công (`reviews.json`) xác nhận cả `5/5 (100%)` câu đều truy xuất đúng đoạn văn chứa nguyên văn đáp án (do `q16` và `q18` trích ở đoạn liền trang có cùng thông tin).*

### Giải thích cấu hình, phân tích lỗi `dev` và lý do chọn `config2`

- **Sự kết hợp giữa phương pháp mới và cải thiện:** Ở tầng retrieval cơ sở, **BM25** và **TF-IDF** được bổ sung làm phương pháp mới bên cạnh **Dense** của Day 6, cho thấy độ chính xác từ khóa vượt trội (`73.3%` so với `20.0%`). Từ đó, **Cải thiện 1 (`config1`)** hợp nhất cả 3 bộ tìm kiếm bằng **Weighted RRF** (`hybrid_search`) đồng thời nâng `budget` từ `800` lên `2400` ký tự và cho phép agent viết lại truy vấn tối đa 2 lượt. **Cải thiện 2 (`config2`)** tiếp tục bổ sung cơ chế **mở rộng chunk liền kề (`expand_neighbors=True`)** kèm `budget=3600` ký tự để xử lý hiện tượng câu trả lời nằm vắt qua ranh giới 2 chunk liên tiếp.
- **Hai lỗi điển hình trên `dev`, nguyên nhân và cách khắc phục:**
  1. **Lỗi 1 — Thiếu bằng chứng & từ chối nhầm do retrieval lệch ngữ nghĩa và budget quá chật (`q01`, `q04`, `q07–q09`, `q11–q13` ở `config0`):**
     - *Hiện tượng:* Ở `q01` (*"What company did Mr. Dursley direct and what product did it make?"*), `config0` dùng `dense_search` nên kéo nhầm chunk `hp-book1-ch1-4_p16_para_2` (cảnh ông Dursley nói chuyện với vợ) thay vì trang 12 (`Grunnings` / `drills`). Do `budget=800` chỉ nhét vừa duy nhất 1 chunk (~762 ký tự) và không có đáp án, agent buộc phải từ chối (`false_abstention` = 8/15 câu trên `config0`).
     - *Cách thử & kết quả:* Chuyển sang `hybrid_search` + nâng `budget=2400` ở `config1` đã sửa dứt điểm toàn bộ 8 ca này, đưa `Context Evidence Rate` từ **20.0% lên 86.7%** và giảm `false_abstention` từ **8 xuống 0**.
  2. **Lỗi 2 — Đứt mạch thông tin ở ranh giới chunk/trang (`q10` ở `config1`):**
     - *Hiện tượng:* Ở `q10` (*"Which secondary school was Harry supposed to attend in September, and what was Aunt Petunia doing to prepare his uniform?"*), `config1` truy xuất được chunk đầu đoạn nhưng phần chi tiết về đồng phục nằm ở chunk kế tiếp bị cắt ngoài top-3 (`context_hit=False`).
     - *Cách thử & kết quả:* Bật `expand_neighbors=True` (tự động nạp thêm chunk `pos+1` và `pos-1` liền trang với trọng số `0.9*score`) cùng `budget=3600` ở `config2` đã nối liền đoạn văn bị cắt, giúp `q10` đạt `context_hit=True` và nâng tỷ lệ tìm trúng bằng chứng trên `dev` lên **14/15 (93.3%)**.
- **Lý do chọn `config2`:** Trên tập `dev`, `config2` đạt tỷ lệ bao phủ bằng chứng cao nhất (**14/15 = 93.3%**), đạt **0** ca từ chối nhầm (`false_abstention = 0`), giữ **100%** từ chối đúng trên câu ngoài phạm vi (`2/2`), trong khi chi phí toàn bộ tập `dev` chỉ tăng thêm `$0.0008` so với `config1` (`$0.0069` vs `$0.0061`).

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **4 Tool retrieval (`MultiToolRetriever`):** `dense_search` (ChromaDB Day 6), `lexical_search` (TF-IDF với fallback BM25), `hybrid_search` (Weighted RRF kết hợp TF-IDF + BM25 + Dense), và `metadata_filter_search` (tự động lọc theo `chapter` hoặc `page`).
  - **Số lượt gọi & điều kiện dừng:** Ở lượt đầu, controller bắt buộc gọi `search` trước khi trả lời (`search_and_answer` / `search_and_abstain`). Vòng lặp dừng khi: (i) agent xuất hành động `answer` với citation hợp lệ, (ii) agent xuất hành động `abstain` khi thiếu bằng chứng, hoặc (iii) chạm trần `max_searches` (`1` ở `config0`, `2` ở `config1`, `3` ở `config2`) — khi vượt quá giới hạn tìm kiếm, hệ thống tự động trả về `abstain`.
  - **Cách chọn context & bảo vệ prompt injection:** Các chunk truy xuất (cùng chunk liền kề ở `config2`) được loại trùng theo `id`, bọc trong thẻ `<untrusted_data citation="[source_id:page:chunk_id]" id="..." source="..." page="...">...</untrusted_data>` trong giới hạn `budget` ký tự, kèm quy tắc hệ thống coi dữ liệu bên trong là văn bản thô, không thực thi chỉ thị.
  - **Xử lý lỗi & kiểm chứng citation:** Hàm `_resolve_citations` xác minh mọi citation phải thuộc `context_ids` thực tế và xuất hiện đúng định dạng trong câu trả lời; nếu không hợp lệ, hệ thống hủy câu trả lời và chuyển sang `abstain` (`"Không đủ bằng chứng để trả lời."`). Lỗi mạng/HTTP 429 được retry có backoff (tối đa 5 lần) hoặc ghi nhận riêng trạng thái `status: "error"` (không bao giờ thay bằng kết quả giả).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Baseline Agent (`config0` / `dev`)** | 17/17 (100%) | TB 1.00 search (2.00 LLM calls) | 8 / 0 / 7 — TB **0.93** (8 câu lỗi `0` do false abstain) | 7/7 (100%) | 7/7 (100%) | 2/2 (100%) | 8/15 (53.3%) |
| **Cải thiện 1 (`config1` / `dev`)** | 17/17 (100%) | TB 1.06 search (2.06 LLM calls) | 0 / 2 / 13 — TB **1.87** (13/15 đạt điểm tối đa 2) | 15/15 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0.0%) |
| **Cải thiện 2 (`config2` / `dev`)** | 17/17 (100%) | TB 1.00 search (2.00 LLM calls) | 0 / 1 / 14 — TB **1.93** (14/15 đạt điểm tối đa 2) | 15/15 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0.0%) |
| **Cấu hình chọn (`config2` / `test`)** | **7/7 (100%)** | **TB 1.00 search (2.00 LLM calls)** | **0 / 0 / 5 — TB 2.00** (`5/5` đạt điểm 2 theo `human_eval_summary.json`) | **5/5 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

### Bảng chi tiết từng câu trên tập `test` (`config2`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q16** | Có evidence (Chapter 1) | `hybrid_search` (1 lượt, 5 chunks sau neighbor expansion, 3,408 chars) $\rightarrow$ lấy được `p16_para_2` | *"The peculiar events reported on the evening television news that made Mr. Dursley anxious included sightings of owls and shooting stars, as well as reports of many funny-looking people in town `[hp-book1-ch1-4:16:hp-book1-ch1-4_p16_para_2]`."* | `answer_grade=2`, `citation_support=[true]`. Nêu đủ 3 sự kiện trên bản tin tối; đoạn trích `p16_para_2` xác nhận nguyên văn. |
| **q17** | Có evidence (Chapter 2) | `hybrid_search` + filter Chapter 2 (1 lượt, 5 chunks, 3,172 chars) $\rightarrow$ neighbor expansion kéo được `p34_para_2` | *"Khi con trăn boa trườn ra khỏi bể, nó đã rít lên với Harry: \"Brazil, here I come. . . . Thanksss, amigo.\" `[hp-book1-ch1-4:34:hp-book1-ch1-4_p34_para_2]`"* | `answer_grade=2`, `citation_support=[true]`. Trích chính xác lời con trăn boa nhờ mở rộng chunk liền kề `p34_para_2`. |
| **q18** | Có evidence (Chapter 3) | `hybrid_search` (1 lượt, 5 chunks, 3,060 chars) $\rightarrow$ neighbor expansion kéo được `p43_para_3` | *"Harry failed in his plan because as he crept across the dark hall toward the front door, he trodden on something big and squashy, which caused him to scream and likely alerted the household `[hp-book1-ch1-4:43:hp-book1-ch1-4_p43_para_3]`."* | `answer_grade=2`, `citation_support=[true]`. Nêu đúng nguyên nhân thất bại (dẫm phải chú Vernon nằm ngủ canh cửa và hét lên). |
| **q19** | Có evidence (Chapter 3) | `hybrid_search` (1 lượt, 5 chunks, 3,313 chars) $\rightarrow$ lấy được `p48_para_0` | *"Uncle Vernon brought a bag of chips for each person and four bananas as rations for their stay at the shack on the rock `[hp-book1-ch1-4:48:hp-book1-ch1-4_p48_para_0]`."* | `answer_grade=2`, `citation_support=[true]`. Đúng chính xác khẩu phần (mỗi người một túi khoai tây chiên và 4 quả chuối). |
| **q20** | Có evidence (Chapter 4) | `hybrid_search` + filter Chapter 4 (1 lượt, 5 chunks, 3,217 chars) $\rightarrow$ lấy được `p56_para_0` | *"Hagrid explained that a 'Muggle' is what wizards call 'nonmagic folk' `[hp-book1-ch1-4:56:hp-book1-ch1-4_p56_para_0]`."* | `answer_grade=2`, `citation_support=[true]`. Giải thích đúng nguyên văn lời Hagrid (`"nonmagic folk"`). |
| **q23** | Thiếu evidence (Book 2) | `hybrid_search` (1 lượt, 5 chunks về Dursley/thư Hogwarts, 3,141 chars) $\rightarrow$ không có thông tin Tom Riddle | *"Không đủ bằng chứng để trả lời."* (không citation) | `abstention_correct=true`. Từ chối chính xác câu hỏi ngoài phạm vi Chapters 1–4. |
| **q24** | Thiếu evidence (Book 7) | `hybrid_search` (1 lượt, 4 chunks về thư Hogwarts/nhà trọ, 2,837 chars) $\rightarrow$ không có thông tin Horcrux | *"Không đủ bằng chứng để trả lời."* (không citation) | `abstention_correct=true`. Từ chối chính xác câu hỏi ngoài phạm vi Chapters 1–4. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - **Đường dẫn dữ liệu & kết quả:**
    - Dữ liệu chuẩn hóa: `data/chunks.json`, `data/questions.json`, vector store `chroma_db/`
    - Kết quả baseline retrieval (`dev` & `test`): `starter/results/dev-dense/`, `starter/results/dev-bm25/`, `starter/results/dev-tfidf/`, `starter/results/dev-hybrid/`, `starter/results/test-dense/`
    - Trace & summary Agent (`dev`): `starter/results/agent-dev/config0/`, `starter/results/agent-dev/config1/`, `starter/results/agent-dev/config2/`
    - Trace, nhãn chấm tay & tổng hợp (`test`): `starter/results/agent-test/traces.json`, `starter/results/agent-test/summary.json`, `starter/results/agent-test/cases.json`, `starter/results/agent-test/reviews.json`, `starter/results/agent-test/human_eval_summary.json`
  - **Lệnh chạy tái lập:**
    ```powershell
    $env:PYTHONUTF8=1
    python -m unittest discover -s tests -v
    python starter/baseline.py --strategy dense --chunks data/chunks.json --questions data/questions.json --split test --output starter/results/test-dense
    python starter/agentic.py --split dev --config config0 --delay 20 --max-requests 40 --skip-existing --output starter/results/agent-dev/config0
    python starter/agentic.py --split dev --config config1 --delay 20 --max-requests 40 --skip-existing --output starter/results/agent-dev/config1
    python starter/agentic.py --split dev --config config2 --delay 20 --max-requests 55 --skip-existing --output starter/results/agent-dev/config2
    python starter/agentic.py --split test --config config2 --delay 20 --max-requests 25 --skip-existing --output starter/results/agent-test
    python starter/agent_metrics.py --cases starter/results/agent-test/cases.json --traces starter/results/agent-test/traces.json --reviews starter/results/agent-test/reviews.json --split test --output starter/results/agent-test/human_eval_summary.json
    ```
  - **Thời gian và chi phí:**
    - Toàn bộ chạy thử nghiệm trên `dev` (3 configs × 17 câu = 51 lượt chạy): tổng thời gian chờ API ~39.7 phút (do `--delay 20` giây/call), tổng token `86,794` prompt + `7,001` completion = **`$0.0172` USD** (`0` lỗi API).
    - Lần đánh giá độc lập trên `test` (`config2`, 7 câu): tổng thời gian `300.8s` (`42.98s`/câu), `14,573` prompt tokens + `986` completion tokens = **`$0.0028` USD** (`0` lỗi API).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Kết quả trước/sau:** So với `config0` (Baseline Agent), `config2` nâng `Context Evidence Rate` trên `dev` từ **20.0% lên 93.3%**, giảm `false_abstention` từ **8/15 xuống 0/15**, nâng điểm câu trả lời trung bình từ **0.93 lên 1.93** trên `dev` và đạt **2.00/2.00** trên `test`, đồng thời giữ **100%** khả năng từ chối câu hỏi ngoài phạm vi.
  - **Ví dụ lỗi điển hình:**
    - *Lỗi ranh giới đoạn văn (`q17`, `q18` trên `test` / `q10` trên `dev`):* Ở `q17`, điểm rank 1–2 của `hybrid_search` dừng ở cảnh tấm kính biến mất (`p34_para_1`), trong khi câu nói của con trăn nằm ở đầu đoạn kế tiếp (`p34_para_2`). Nhờ `expand_neighbors=True`, `p34_para_2` được nạp tự động vào context và agent trả lời chính xác 100%.
    - *Lỗi lệch ID gold tự động (`q16`, `q18` trên `test`):* Máy chấm tự động báo `context_hit=false` vì gold gốc gắn ở đoạn liền trang, nhưng kiểm chứng thủ công cho thấy đoạn chunk mà agent trích dẫn chứa đầy đủ bằng chứng chính xác.
  - **Đánh đổi & giới hạn:**
    1. *Đánh đổi chất lượng — độ trễ — chi phí:* Tăng budget từ `800` lên `3600` ký tự và mở rộng neighbor làm tăng ~73% lượng prompt tokens (`20.5K -> 35.7K` trên `dev`), nhưng chi phí thực tế vẫn cực thấp (`<$0.007` cho cả tập `dev`). Độ trễ ~43–47s/câu chủ yếu do `--delay 20` giây/call để tránh vượt ngưỡng 15 RPM của gói miễn phí.
    2. *Giới hạn:* Truy vấn mở rộng neighbor chỉ xét `±1` đoạn liền kề trong cùng/liền trang nên chưa tối ưu cho câu hỏi tổng hợp nhiều chương xa nhau; đồng thời hệ thống chưa áp dụng reranker cross-encoder chuyên biệt.
    
## Thử thách nâng cao (nếu thực hiện)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - **Tài liệu & phiên bản:** *NASA Systems Engineering Handbook* (`NASA/SP-2007-6105 Rev2`).
  - **Nguồn truy cập công khai:** `https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf`
  - **Quyền sử dụng:** Public Domain (ấn phẩm của Chính phủ Hoa Kỳ — NASA, được phép sử dụng, trích xuất và tái tạo tự do cho mục đích học thuật/nghiên cứu).
  - **Số trang:** **297 trang** đầy đủ từ trang bìa đến hết danh mục tài liệu tham khảo.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - **Phạm vi index (`data_adv/index_stats.json`):** Index toàn bộ **297 / 297 trang** (`0` trang bị bỏ qua), tạo thành **338 chunks** lưu tại `data_adv/chunks_adv.json` và collection ChromaDB `nasa_se_adv` (`sentence-transformers/all-MiniLM-L6-v2`, cosine similarity), thời gian index `345.48s`.
  - **Cách xử lý từng loại nội dung (`scripts/index_multimodal.py` & `starter/agentic_adv.py`):**
    - *Text:* Tách theo đoạn văn trên toàn bộ `297` trang, lưu kèm metadata `source_id` (`nasa-se-rev2`), số trang PDF (`page`) và tọa độ `bbox`.
    - *Bảng (`38` trang có bảng):* Trích xuất bằng `pymupdf` (`page.find_tables()`) sang **Markdown table** giữ nguyên cấu trúc hàng/cột và `bbox`, đồng thời render PNG trang gốc (`dpi=150`) để kiểm chứng trực quan.
    - *Ảnh, sơ đồ và biểu đồ (`80` trang có đồ họa):* Phát hiện các trang chứa ảnh raster ($\ge 120 \times 120$ px) hoặc mật độ nét vẽ vector cao (`len(drawings) > 50`), render PNG lưu tại `data_adv/pages/page_XXX.png`. Khi truy vấn trúng trang có `has_visual=True`, agent đính kèm tối đa 3 ảnh PNG trang gốc qua `inlineData` (`image/png`) trực tiếp vào `gemini-3.1-flash-lite` để đọc trục, đơn vị, chú giải và luồng sơ đồ.
  - **Phần chưa xử lý / hạn chế lúc index:** Các trang chỉ chứa sơ đồ/đồ thị vector tối giản có ít hơn 50 nét vẽ (`len(drawings) <= 50`, cụ thể là Page 61 và Page 81) chưa được tự động gắn cờ `has_visual=True` nên chưa render PNG đính kèm lúc suy luận.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - **Bộ benchmark (`data_adv/questions_adv.json`, 48 câu, tách biệt hoàn toàn `dev` và `test`):**
    - **`dev` (18 câu):** 15 câu có đáp án phủ đủ 5 nhóm (4 Bảng, 4 Sơ đồ, 2 Biểu đồ/thang đo, 2 Multi-hop, 3 Text) + 3 câu thiếu bằng chứng (`adv_dev_01`–`adv_dev_18`).
    - **`test` (30 câu):** 24 câu có đáp án chia đúng định mức (6 Bảng, 5 Ảnh/sơ đồ, 5 Biểu đồ, 4 Multi-hop, 4 Text) + 6 câu thiếu bằng chứng (`adv_test_01`–`adv_test_30`).
  - **Schema evidence:** Mỗi bản ghi gồm `id`, `split`, `category`, `answerable`, `question`, `answer` (hoặc chuỗi từ chối chuẩn `"Không đủ bằng chứng để trả lời."`), `gold_pages` (số trang PDF gốc) và `evidence` (mô tả bảng/hàng/cột/nhãn trục/đơn vị cụ thể).
  - **Model & cấu hình đã chốt trên `dev`:** Hybrid RRF retrieval (`1.5 * TF-IDF + 1.2 * BM25 + 1.0 * Dense` + page-anchor boost, `top_k=5` + sibling chunks cùng trang) kết hợp multimodal vision (`gemini-3.1-flash-lite`, `temperature=0.0`, đính kèm tối đa 3 page PNG). Trên `dev` đạt **18 / 18 (100%)** (`15/15` có đáp án, `3/3` từ chối đúng, `0` lỗi, chi phí `$0.016931` USD, latency trung bình `30,895.2 ms`).

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6 / 6 | 100.0% | Tra cứu chính xác hàng, cột, so sánh giữa các loại sứ mệnh Type A–F và bảng quy trình AS9100/V&V (`adv_test_01`–`06`). |
| Ảnh, sơ đồ | 4 / 5 | 80.0% | Đạt 4/5 (`adv_test_07`–`09, 11`); lỗi 1 câu (`adv_test_10`, Page 61) do sơ đồ vector ít nét vẽ nên chưa gắn cờ render PNG. |
| Biểu đồ | 4 / 5 | 80.0% | Đạt 4/5 (`adv_test_12, 14`–`16`: đọc đúng `90%`, `500–1000x`, thang TRL 1–9, ma trận màu TRL); lỗi 1 câu (`adv_test_13`, Page 81) do đồ thị vector chưa render PNG. |
| Multi-hop | 4 / 4 | 100.0% | Tổng hợp chính xác bằng chứng từ 2 trang khác nhau (`19+28`, `48+49`, `149+152`, `17+171`) (`adv_test_17`–`20`). |
| Text | 4 / 4 | 100.0% | Trích xuất chính xác điều kiện NPR, baselines, phân biệt `Shall` vs `Will`, và hoạt động Phase E (`adv_test_21`–`24`). |
| Thiếu bằng chứng | 6 / 6 | 100.0% | Từ chối an toàn toàn bộ 6 câu hỏi ngoài phạm vi tài liệu (Artemis II, Perseverance, Saturn V, Dragon 2, Europa Clipper, Starship) (`adv_test_25`–`30`). |
| **Toàn bộ** | **28 / 30** | **93.3%** | **Vượt ngưỡng yêu cầu $\ge 90\%$ (`27/30`)** với `0` lỗi hệ thống/API. |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:**
  - **File trace & nhãn đối chiếu từng câu:** `starter/results/adv-dev/` (`traces.json`, `summary.json`) và `starter/results/adv-test/` (`traces.json`, `summary.json`, `reviews_adv.json` ghi nhãn `passed` và lý do đối chiếu PDF gốc cho từng câu trong 30 câu `test`, `group_summary.json`).
  - **Thời gian, chi phí & lỗi (`test`, 30 câu):** Latency trung bình `35,932.4 ms`/câu (đã bao gồm `--delay 20` giây/request để tuân thủ giới hạn Free Tier), tổng token `175,984` prompt + `3,113` completion = **`$0.028265` USD**; **`0` lỗi hệ thống / timeout / API**.
  - **Phân tích 2 lỗi trên `test`:** Cả 2 câu không đạt (`adv_test_10` và `adv_test_13`) đều truy xuất trúng trang nguồn (`page_hit=True` tại Page 61 và Page 81), nhưng do ngưỡng phát hiện đồ họa vector `len(drawings) > 50` bỏ sót 2 hình vẽ vector tối giản này lúc index nên `image_path` rỗng, khiến agent từ chối an toàn (`false abstention`).
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - **Giới hạn:** Việc lọc trang trực quan dựa vào ngưỡng số nét vẽ (`> 50` drawings) có thể bỏ sót các biểu đồ/sơ đồ vector đơn giản; có thể khắc phục ở bước index bằng cách render PNG cho mọi trang có tiêu đề `"Figure "` ở đầu dòng.
  - **Đường dẫn benchmark & script:**
    - Bộ câu hỏi & đáp án chuẩn: `data_adv/questions_adv.json` (`18 dev` + `30 test`)
    - Chunks & thống kê index: `data_adv/chunks_adv.json`, `data_adv/index_stats.json`
    - Script index & multimodal agent: `scripts/index_multimodal.py`, `starter/agentic_adv.py`
  - **Lệnh tái lập:**
    ```powershell
    $PY = "..\K5-L3B-DAY06-2A202603820-DataIngestion\.venv\Scripts\python.exe"
    New-Item -ItemType Directory -Force -Path data_adv | Out-Null
    curl.exe -L "[https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf](https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf)" -o "data_adv/report.pdf"
    & $PY scripts/index_multimodal.py
    & $PY starter/agentic_adv.py --split dev --delay 20 --max-requests 25 --skip-existing --output starter/results/adv-dev
    & $PY starter/agentic_adv.py --split test --delay 20 --max-requests 35 --skip-existing --output starter/results/adv-test
    ```
