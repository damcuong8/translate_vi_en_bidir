# English ↔ Vietnamese Transformer (FSDP + MoE)

Hệ thống dịch máy Anh–Việt/Việt–Anh xây dựng trên Transformer tùy biến, hỗ trợ FSDP, MoE, torch.compile, AMP và beam search. Pipeline làm việc trực tiếp với tập dữ liệu đã token hóa ở định dạng Hugging Face `load_from_disk`.

## Kiến trúc & tính năng chính
- **Transformer encoder/decoder tùy biến:** 12 encoder layer, 9 decoder layer, hidden 512, head dim 64, 8 heads. RMSNorm, RoPE, label smoothing 0.06.
- **Mixture-of-Experts (MoE):** 16 expert, top‑2; hỗ trợ FSDP MoE hoặc DeepSpeed MoE, shared embedding và tie weights.
- **Tăng tốc:** AMP (fp16/bf16), optional FlashAttention/SDPA kernel, torch.compile, gradient accumulation, gradient clipping.
- **Huấn luyện phân tán:** FSDP (mặc định), DDP hoặc DeepSpeed Zero-2; checkpoint có thể ở dạng DCP và được tự xử lý/convert.
- **Scheduler:** Cosine decay + warmup, AdamW; log với W&B.
- **Sinh chuỗi:** Greedy hoặc beam search (length penalty) qua `Transformer.generate`.

## Yêu cầu
```
pip install -r requirements.txt
```
Thêm `sentencepiece`, `sacrebleu`, `comet-ml` nếu cần đánh giá đầy đủ.

## Dữ liệu & tokenizer
- Tokenizer Hugging Face tại `tokenizer_path` (mặc định `./Tokenizer_ENVI/`), chứa token ngôn ngữ `__eng__`, `__vie__`.
- Tập dữ liệu dùng `load_from_disk`, mỗi mẫu cần các trường:
  - `input_ids_en`, `input_ids_vi`, `en`, `vi`
- Lớp `BidirectionalDataset` nhân đôi mẫu theo hai hướng và thêm lang-token ở đầu chuỗi; lọc mẫu dài hơn `max_seq_len` (mặc định 149) và cache tại `/tmp/translate_vi_en_cache`.
- `Collator` padding về bội số 8, trả về `src_input_ids`, `tgt_input_ids`, mask và `labels` cho LM loss.

## Cấu hình huấn luyện
- File mặc định: `train_config.json`. Các khóa quan trọng:
  - `use_single_gpu` / `use_fsdp` / `use_ddp` / `use_deepspeed`
  - Batch size per GPU (`train_batch_size`, …), `gradient_accumulation_steps`
  - Optimizer/scheduler (AdamW + WarmupCosine), `use_amp`, `amp_dtype`
  - FSDP: `sharding_strategy`, `use_mixed_precision`, `cpu_offload`, `use_torch_compile`
  - Đường dẫn dữ liệu/tokenizer/checkpoint, `lang_token_map`
- Nếu muốn cấu hình nhanh cho Kaggle, xem hàm `get_kaggle_config()` trong `config.py`.

## Chạy huấn luyện
`train.py` tự chọn chế độ theo config/flag.

```bash
# FSDP nhiều GPU (khuyến nghị)
torchrun --nproc_per_node=4 train.py --config train_config.json

# Một GPU
python train.py --config train_config.json --single_gpu

# DeepSpeed Zero-2
deepspeed --num_gpus=4 train.py --config train_config.json --ds_config deepspeed_config.json

# Resume
torchrun --nproc_per_node=4 train.py --config train_config.json --resume_from_checkpoint checkpoints/checkpoint-xxxxx
```

Checkpoint được lưu định kỳ (`save_steps`) và cuối mỗi epoch vào `checkpoint_path`. `checkpoint_utils.py` hỗ trợ lưu/khôi phục (kể cả DCP).

## Đánh giá & suy luận hàng loạt
`evaluate.py` tải mô hình, tự chuyển đổi DCP nếu cần, sinh bản dịch bằng `model.generate`, tính BLEU/chrF++ và (tùy chọn) COMET, lưu kết quả vào `evaluation_results/`.

Ví dụ:
```bash
python evaluate.py \
  --config train_config.json \
  --checkpoint checkpoints/checkpoint-131410 \
  --dataset_path ./flores_tokenized/flores_devtest \
  --beam_size 5 \
  --length_penalty 1.0 \
  --bidirectional True
```
Kết quả gồm: CSV predictions/references/sources và file JSON/summary với BLEU, chrF++, COMET (nếu tải được).

## Suy luận nhanh (gợi ý)
- Dùng `evaluate.load_model_and_tokenizer` để tải checkpoint và tokenizer, sau đó gọi `model.generate` với `tgt_start_ids = [[bos, lang_token]]`.
- Script `translate.py` là phiên bản cũ (không theo mô hình hiện tại); ưu tiên dùng pipeline trong `evaluate.py`.

## Thư mục chính
- `train.py`: bộ chọn chế độ huấn luyện (single/FSDP/DDP/DeepSpeed).
- `train_fsdp.py`, `train_ddp.py`, `train_deepspeed.py`, `train_single.py`: logic huấn luyện cụ thể.
- `model.py`: định nghĩa Transformer + MoE, generate (greedy/beam).
- `dataset.py`: `BidirectionalDataset`, `Collator`.
- `checkpoint_utils.py`: lưu/khôi phục, hỗ trợ DCP.
- `evaluate.py`: suy luận + tính BLEU/chrF++/COMET.
- `train_config.json`: cấu hình mẫu.

