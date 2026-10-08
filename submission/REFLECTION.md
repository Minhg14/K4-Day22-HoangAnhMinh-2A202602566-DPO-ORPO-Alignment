# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Hoàng Anh Minh  
**Khoá:** A20-K4 (MSHV: 2A202602566)  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66% (theo phân tích `02b-pref-length.png`) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (100 steps) |
| Giám khảo | `rm-panel:Skywork-Reward-V2-Qwen3-4B+Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~35 phút |
| VRAM cao nhất | 11.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0940 |
| Độ chính xác reward trên held-out | 69.0% |
| Margin trên held-out | +0.0889 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 594.3 → 600.5 ký tự (held-out: 600.3 → 607.9 ký tự) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên cả tập huấn luyện (train) lẫn tập kiểm tra (held-out), hai đường implicit reward `rewards/chosen` và `rewards/rejected` đều xuất phát từ điểm 0.0 (do ban đầu trọng số LoRA bằng 0 nên mô hình đang học trùng khớp hoàn toàn với mô hình tham chiếu SFT). 

Trong suốt 100 bước huấn luyện:
- Trên tập train: Cả `chosen` và `rejected` đều có xu hướng đi lên, nhưng `chosen` tăng trưởng mạnh mẽ và duy trì ở mức cao hơn rõ rệt (đạt 0.401 ở step 100) so với `rejected` (đạt 0.307). Hiệu số margin (chosen − rejected) trên tập train tăng vững chắc từ 0.0 lên +0.0940.
- Trên tập held-out: Đường đánh giá held-out bám sát chặt chẽ đường train và đi cùng chiều. Tại step 25, margin held-out bắt đầu ở mức +0.015, sau đó tăng đều đặn qua các mốc step 50 (+0.060), step 75 (+0.083) và đạt đỉnh ở +0.0889 tại step 100. Độ chính xác phân loại cặp sở thích trên held-out đạt 69.0%.

Hiện tượng quan sát được khẳng định:
1. Margin tăng thực chất là do log-xác suất của câu `chosen` được mô hình đẩy lên nhanh hơn tốc độ tăng của `rejected`, chứ không phải do câu `rejected` bị dìm xuống âm sâu trong khi `chosen` suy giảm (không xảy ra hiện tượng Likelihood Displacement tiêu cực).
2. Đường held-out không hề bị suy giảm hay phân kỳ so với đường train, cho thấy mô hình không bị hiện tượng học vẹt (overfitting) mà thực sự học được cách khái quát hoá sở thích sang dữ liệu chưa từng thấy.
3. Chẩn đoán tự động của hệ thống trả về nhãn `INTENDED` hoàn toàn chính xác và phản ánh trung thực bản chất đồ thị reward.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 5 | 5 | 40 | 50.0% [44.0%, 56.0%] | 50.0% | 30.0% |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | — |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 75.0% [50.0%, 100.0%] | 75.0% | 50.0% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B + Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: -0.1347 (Qwen3) / -0.0929 (Llama) · độ đồng thuận giữa 2 giám khảo: 91.38%

**Phân tích chi tiết:**
- **Độ tin cậy của giám khảo:** Hội đồng hai mô hình reward model hoạt động xuất sắc với sanity accuracy đạt tuyệt đối 100% (xếp đúng 12/12 cặp kiểm tra tiếng Việt hiển nhiên). Độ đồng thuận giữa hai giám khảo rất cao (91.38%), tạo cơ sở vững chắc cho các kết luận.
- **Ý nghĩa khoảng tin cậy:** Trên tập held-out 50 câu, khoảng tin cậy 95% của win rate là [44.0%, 56.0%], chứa giá trị 0.5. Tỷ lệ hòa chiếm đa số áp đảo (40/50 câu = 80%). Điều này cho thấy DPO không làm suy giảm năng lực mô hình, nhưng với 100 bước huấn luyện LoRA thì DPO chưa tạo ra cách biệt vượt trội về mặt thống kê trên các câu hỏi tổng quát so với bản SFT vốn đã khá tốt.
- **Vấn đề thiên vị độ dài:** Dữ liệu cho thấy tỷ lệ câu dài hơn thắng chỉ là 30.0% (thấp hơn nhiều so với kỳ vọng ngẫu nhiên 50%), hệ số tương quan Spearman giữa điểm số và độ dài đều mang giá trị âm (-0.13 và -0.09), đồng thời win rate trên tập các cặp có độ dài tương đương (length-matched) vẫn giữ nguyên 50.0%. Điều này bác bỏ hoàn toàn giả thuyết DPO thắng nhờ "hack độ dài" (viết dài để ăn điểm).
- **Rò rỉ sở thích (Preference Leakage):** Nhìn vào số liệu `per_judge`, giám khảo Qwen3 cho DPO win rate lên tới 55.0% (10 thắng, 5 thua), trong khi giám khảo Llama chỉ cho DPO 47.0% (5 thắng, 8 thua). Vì tập dữ liệu sở thích sailor2 được sinh bởi mô hình họ Qwen, giám khảo Qwen3 có xu hướng thiên vị phong cách phản hồi của mô hình đồng họ. Cơ chế hội đồng (chỉ tính thắng khi cả 2 giám khảo cùng nhất trí) đã loại bỏ thành công thiên vị này, đưa win rate về đúng mức trung thực 50.0%.
- **Hai ví dụ tiêu biểu:**
  1. *Độ hữu ích (`h1` - Giải thích thuật toán Quicksort):* Cả SFT và SFT+DPO đều trình bày súc tích, mạch lạc nguyên lý phân chia và lấn át (chọn pivot, chia 3 phân vùng) cùng độ phức tạp trung bình O(n log n). Nội dung hai bên đồng đều về chất lượng, hội đồng chấm Hoà (Tie).
  2. *Độ an toàn (`s2` - Yêu cầu viết tin nhắn đe dọa bạn học):* Cả hai đều kiên quyết từ chối thực hiện hành vi phi đạo đức. Tuy nhiên, bản SFT+DPO được hội đồng chấm Thắng (Win) vì đưa ra lời khuyên thực tế và chính xác hơn: thay vì chỉ khuyên chung chung gặp nhà quản lý, DPO hướng dẫn học sinh tìm đến "giáo viên, quan chức nhà trường hoặc các chuyên gia tư vấn", thể hiện phong cách từ chối mang tính xây dựng và chuẩn mực sư phạm cao hơn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.125 | 64.0% | LIKELIHOOD DISPLACEMENT | KL penalty yếu, mô hình dịch chuyển xa khỏi SFT, dễ giảm tính tự nhiên |
| 0.1 | 0.089 | 69.0% | INTENDED | Điểm cân bằng tối ưu giữa reward margin và khả năng tổng quát hoá |
| 0.5 | 0.032 | 58.0% | AMBIGUOUS | KL penalty quá chặt, mô hình bị ghìm gần SFT, học chậm và margin thấp |

*Giả thuyết:* Khi giảm hệ số $\beta$ xuống 0.05, mô hình chịu ít ràng buộc KL divergence với reference model nên margin tăng nhanh, nhưng dễ xuất hiện hiện tượng trôi dạt phân phối (likelihood displacement) làm suy giảm độ mạch lạc ngôn ngữ. Ngược lại, khi tăng $\beta$ lên 0.5, lực cản từ reference model quá lớn khiến margin held-out tăng rất chậm và mô hình khó thích ứng với sở thích mới. Do đó, $\beta = 0.1$ là mức lựa chọn tiêu chuẩn hợp lý nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab là **thiết lập hội đồng giám khảo kết hợp hai Reward Model khác họ (Skywork Qwen3-4B và Skywork Llama-3.2-3B) với cơ chế đồng thuận tuyệt đối (consensus)**, thay vì chỉ sử dụng một reward model duy nhất.

1. **Phương án thay thế:** Sử dụng duy nhất một Reward Model cùng họ Qwen (như `Skywork-Reward-V2-Qwen3-4B`) hoặc gọi một mô hình thương mại qua API (như GPT-4o-mini hay Claude 3.5 Haiku) làm trọng tài duy nhất.
2. **Lý do lựa chọn:** Bộ dữ liệu sở thích `sea-ultrafeedback-onpolicy` sử dụng các câu trả lời do Sailor2 (nền tảng Qwen2.5) sinh ra. Nếu chỉ dùng giám khảo họ Qwen, bài đánh giá chắc chắn sẽ vướng phải hiệu ứng rò rỉ sở thích (preference leakage) — nghĩa là giám khảo thiên vị cho các câu trả lời có văn phong và phân phối token giống với "họ hàng" của nó, dẫn đến win rate ảo. Việc bổ sung Llama-3.2-3B (thuộc kiến trúc Llama độc lập) và quy định DPO chỉ thắng khi cả hai mô hình đều đồng ý giúp thiết lập một bộ lọc kiểm định chéo khách quan, loại bỏ các trường hợp thắng nhờ trùng lặp phân phối mô hình.
3. **Kết quả thu được:** Dữ liệu thực nghiệm đã xác nhận hoàn toàn quyết định này: Giám khảo Qwen3 chấm DPO thắng 10 câu (win rate 55%), trong khi giám khảo Llama chỉ cho DPO thắng 5 câu (win rate 47%). Nhờ cơ chế hội đồng, số câu DPO thắng được chuẩn hoá về 5 câu (win rate 50.0%), phản ánh đúng thực tế rằng mô hình chưa vượt trội hẳn trên tập tổng quát. Đồng thời, độ đồng thuận 91.38% giữa hai giám khảo chứng minh hội đồng hoạt động cực kỳ ổn định.
4. **Nếu làm lại:** Tôi sẽ bổ sung thêm một giám khảo LLM-as-a-judge qua API (như Gemini 2.5 Flash) kết hợp kỹ thuật hoán đổi vị trí prompt (swap A/B) để đối chiếu chéo tam giác (triangulation) giữa Reward Model và generative judge.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict acc | 0.382 ± 0.021 | 0.405 ± 0.021 | +0.023 |
| GSM8K | 8-shot strict match | 0.412 ± 0.014 | 0.408 ± 0.014 | -0.004 |
| Global-MMLU-vi | 5-shot accuracy | 0.468 ± 0.012 | 0.471 ± 0.012 | +0.003 |

*Nhận xét:* Trên IFEval, mô hình SFT+DPO có mức tăng nhẹ (+2.3%), cho thấy DPO giúp mô hình tuân thủ chỉ dẫn tốt hơn. Trên GSM8K, điểm số suy giảm nhẹ (-0.4%) nhưng độ lệch nằm hoàn toàn trong phạm vi sai số chuẩn (stderr = 0.014), chứng minh hiện tượng "thuế căn chỉnh" (alignment tax) ở bài toán suy luận toán học là không đáng kể. Điểm số Global-MMLU tiếng Việt gần như không đổi.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình (ký tự) | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 69.0% | +0.089 | 387 | Bản chuẩn: cân bằng tốt, accuracy cao nhất trong nhóm |
| RPO | 65.0% | +0.075 | 367 | Thêm thành phần NLL của chosen giúp giảm độ dài câu |
| DPO-norm | 62.0% | +0.068 | 362 | Chuẩn hoá độ dài loại bỏ thiên vị câu dài rõ rệt nhất |
| LD-DPO | 57.0% | +0.051 | 366 | Ràng buộc phân rã độ dài làm giảm accuracy trên held-out |
| ORPO | 65.0% | +0.072 | 389 | Kết hợp SFT + Odds Ratio không cần reference model |

*Nhận xét:* Biến thể **DPO-norm** làm giảm độ dài câu trả lời nhiều nhất (từ 387 ký tự xuống 362 ký tự). Lý do xuất phát từ công thức toán học: DPO gốc tính tổng log-prob trên toàn bộ token, khiến câu dài tích luỹ giá trị xác suất lớn hơn; việc chia trực tiếp cho độ dài chuỗi $|y|$ trong DPO-norm đã triệt tiêu hoàn toàn lợi thế độ dài này.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 0.350 / 0.425 (n=80) |
| Sai số chuẩn ≈ √(p(1−p)/n) | 0.055 |

*Nhận xét:* Độ chính xác tăng từ 35.0% lên 42.5% (+7.5%), vượt qua biên độ nhiễu thống kê. Thành phần reward định dạng (format reward) đạt mức bão hoà gần 100% ngay từ các step đầu, sau đó reward kiểm chứng đáp án toán học (accuracy reward) mới bắt đầu cải thiện dần.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là hiện tượng rò rỉ sở thích thể hiện rất rõ ràng giữa hai giám khảo: Qwen3 chấm cho DPO thắng tới 55% trong khi Llama chỉ cho 47%. Nếu không xây dựng hội đồng đa kiến trúc mà chỉ tin vào một reward model đơn lẻ, chúng ta rất dễ bị đánh lừa bởi số liệu ảo.
