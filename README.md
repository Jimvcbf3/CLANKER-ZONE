"# CLANKER-ZONE" 


Verified 4-bit OCR runner (Windows/WSL2, RTX 4060 8GB)

Quick start (WSL2):

1) Activate venv and set model dir

source ~/venvs/dots/bin/activate
export MODEL_DIR=/root/models/dots-ocr-4bit

2) Copy image from Windows (optional if already at /root/test.png)

cp /mnt/c/ai-auto/test.png /root/test.png

3) Run with VRAM logging

( nvidia-smi -l 1 > ~/nvidia.log & echo  > /tmp/nvpid )
python ~/CLANKER-ZONE-REPO/dots_ocr_run_4bit.py   --model    --image /root/test.png   --prompt "Read all visible text and return as plain text."   --max-new 256   --debug | tee ~/ocr_out.txt
kill  2>/dev/null || true
tail -n 20 ~/nvidia.log

Flags
- --debug: when present, prints debug lines (expected vision_len, expanded image token count, input_len and output_ids.shape, and one-line prefix/new decodes) then the final OCR text. Default off prints only final OCR text.

Notes
- SDPA path; no flash_attn required on Windows.
- Images auto downscale to ~800k pixels (~1k visual tokens) for 8GB.
- Image-token expansion: the single <|imgpad|> placeholder is expanded to the exact number of visual tokens computed from image_grid_thw and spatial_merge_size.
- Anti-loop params: do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.15, min_new_tokens=1.
- Observed VRAM (RTX 4060 8GB): typically ~4–7 GB depending on image size and max_new.
