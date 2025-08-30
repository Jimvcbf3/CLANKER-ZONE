# dots.ocr local + 4-bit runners (Windows)

## Hardware
- RTX 4060 Laptop (8GB VRAM), Ryzen 7 7435HS, 64GB RAM

## Environment (tested)
- Python: 3.13.x
- Torch: 2.6.0+cu124
- Transformers: 4.55.2
- Accelerate: 1.10.0
- Bitsandbytes: 0.47.0 (Windows wheel)
- OS: Windows 11

## Repro
```bat
py -u C:\ai-auto\dump_config.py --model C:\ai-auto\dots_ocr > logs\dump_config_output.txt
py -u C:\ai-auto\dots_ocr_run.py --model C:\ai-auto\dots_ocr --image C:\ai-auto\test.png --prompt "Read all visible text and return as plain text." --max-new 256 --dtype bf16 > logs\runner_output_working.txt
py -u C:\ai-auto\dots_ocr_run_4bit.py --image C:\ai-auto\test.png --prompt "Read all visible text and return as plain text." --max-new 256 > logs\runner_output_4bit.txt