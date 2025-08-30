# dots_ocr_run.py — keep working parts; add VRAM-safe auto-resize
# Usage:
#   py -u C:\ai-auto\dots_ocr_run.py --model C:\ai-auto\dots_ocr --image C:\ai-auto\test.png --prompt "Read all visible text and return as plain text." --max-new 256 --dtype bf16
# Optional:
#   --max-vis-tokens 700   (cap vision tokens after merge; lower if you still OOM)

import os, argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=256, dest="max_new")
    ap.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    # NEW: safety cap for visual tokens after spatial merge (good default for ~8GB VRAM)
    ap.add_argument("--max-vis-tokens", type=int, default=700, dest="max_vis")
    return ap.parse_args()

def pick_dtype(device, flag):
    if device == "cuda":
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[flag]
    return torch.float32

def nearest_multiple(n, k):
    q = n / k
    r = round(q)
    m = max(k, int(r * k))
    if m <= 0: m = k
    return m

def manual_bchw(img: Image.Image, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,C
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise SystemExit(f"[ERR] image must be RGB, got {arr.shape}")
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # 1,C,H,W
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1,3,1,1)
    return (t - mean_t) / std_t  # float32 CPU

def load_model(model_dir, device, dtype):
    print(f"[load] processor from {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    print("[load] model (this may take a bit)…")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[load] model ready (weight dtype={model.dtype})")
    return model, processor

def get_patch_and_merge(model):
    patch = None
    sms = getattr(getattr(model, "vision_tower", None), "spatial_merge_size", None)
    try:
        patch = getattr(model.vision_tower.patchifier, "patch_size", None)
    except Exception:
        patch = None
    if patch is None:
        try:
            ks = model.vision_tower.patchifier.proj.kernel_size
            patch = ks[0] if isinstance(ks, (tuple, list)) else int(ks)
        except Exception:
            patch = 14
    if sms is None:
        sms = 2
    return int(patch), int(sms)

def resolve_image_token_id(tokenizer, model):
    # Prefer numeric ids from config
    for key in ("image_token_index", "image_token_id"):
        if hasattr(model.config, key):
            val = getattr(model.config, key)
            if isinstance(val, int) and val >= 0:
                return val
    # Try configured strings
    candidate_strs = []
    for key in ("image_token", "visual_token"):
        v = getattr(model.config, key, None)
        if isinstance(v, str) and v:
            candidate_strs.append(v)
    candidate_strs += ["<image>", "<img>", "<image_token>"]
    for tok_str in candidate_strs:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            return tid
    # Register a default if needed
    tok_str = "<image>"
    tokenizer.add_special_tokens({"additional_special_tokens": [tok_str]})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    return tokenizer.convert_tokens_to_ids(tok_str)

def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype  = pick_dtype(device, args.dtype)
    print(f"[boot] device={device} dtype={dtype}")

    model_dir = os.path.abspath(args.model)
    image_fp  = os.path.abspath(args.image)
    if not os.path.isdir(model_dir):
        raise SystemExit(f"[ERR] model folder not found: {model_dir}")
    if not os.path.isfile(image_fp):
        raise SystemExit(f"[ERR] image not found: {image_fp}")

    model, processor = load_model(model_dir, device, dtype)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)

    # ---- load image (unchanged) ----
    img = Image.open(image_fp).convert("RGB")
    print(f"[img ] loaded: {image_fp}")

    # ---- factor alignment (unchanged) ----
    patch, sms = get_patch_and_merge(model)
    factor = patch * sms
    H0, W0 = img.height, img.width
    H1 = nearest_multiple(H0, factor)
    W1 = nearest_multiple(W0, factor)
    if (H1, W1) != (H0, W0):
        print(f"[resize] ({H0}x{W0}) -> ({H1}x{W1}) (factor={factor}=patch{patch}*merge{sms})")
        img = img.resize((W1, H1), Image.BICUBIC)

    # ---- NEW: VRAM-safe auto-shrink if visual tokens exceed cap ----
    # visual tokens after merge = (H_tokens * W_tokens) // (sms*sms)
    def tokens_after_merge(height, width):
        Ht = (height // patch)
        Wt = (width  // patch)
        return (Ht * Wt) // (sms * sms)

    vis_now = tokens_after_merge(img.height, img.width)
    if vis_now > args.max_vis:
        # scale height/width by sqrt(target/current) and re-align to factor
        scale = (args.max_vis / max(1, vis_now)) ** 0.5
        newH = max(factor, int(round(img.height * scale)))
        newW = max(factor, int(round(img.width  * scale)))
        newH = nearest_multiple(newH, factor)
        newW = nearest_multiple(newW, factor)
        # Guard: never upscale accidentally
        newH = min(newH, img.height)
        newW = min(newW, img.width)
        if (newH, newW) != (img.height, img.width):
            print(f"[vram] vis_tokens {vis_now} > cap {args.max_vis} -> downscale to {newW}x{newH}")
            img = img.resize((newW, newH), Image.BICUBIC)
            # Recompute for logs
            vis_now = tokens_after_merge(img.height, img.width)
            print(f"[vram] new vis_tokens={vis_now}")

    # ---- BCHW (unchanged) ----
    mean = getattr(processor, "image_mean", getattr(getattr(processor, "image_processor", None), "image_mean", [0.5,0.5,0.5]))
    std  = getattr(processor, "image_std",  getattr(getattr(processor, "image_processor", None), "image_std",  [0.5,0.5,0.5]))
    pixel_values = manual_bchw(img, tuple(mean), tuple(std)).to(device=device, dtype=dtype)

    # ---- grid_thw (unchanged) ----
    _, _, H, W = pixel_values.shape
    H_tokens = H // patch
    W_tokens = W // patch
    if (H_tokens % sms != 0) or (W_tokens % sms != 0):
        raise SystemExit(f"[ERR] token grid not divisible by merge size: H_tokens={H_tokens}, W_tokens={W_tokens}, merge={sms}")
    T = 1
    grid_thw = torch.tensor([[T, H_tokens, W_tokens]], dtype=torch.int32, device=device)

    # ---- input_ids: repeat image token to match visual tokens (unchanged) ----
    img_tok_id = resolve_image_token_id(tok, model)
    vis_tokens = (H_tokens * W_tokens) // (sms * sms)
    if vis_tokens < 1:
        raise SystemExit(f"[ERR] computed vis_tokens < 1 (H_tokens={H_tokens}, W_tokens={W_tokens}, merge={sms})")
    image_ids = torch.full((1, vis_tokens), img_tok_id, device=device, dtype=torch.long)

    prompt_ids = tok(args.prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    if prompt_ids.dtype != torch.long:
        prompt_ids = prompt_ids.to(torch.long)
    input_ids  = torch.cat([image_ids, prompt_ids], dim=1)
    attn_mask  = torch.ones_like(input_ids, dtype=torch.long, device=device)

    print(f"[shapes] input_ids={tuple(input_ids.shape)} pixel_values={tuple(pixel_values.shape)} grid={tuple(grid_thw.shape)}")

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        max_new_tokens=args.max_new,
        do_sample=False,
        temperature=1e-6,
    )
    if getattr(tok, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tok.pad_token_id
    if getattr(tok, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tok.eos_token_id

    with (torch.autocast(device_type="cuda", dtype=dtype) if device=="cuda" and dtype!=torch.float32 else torch.no_grad()):
        out_ids = model.generate(**gen_kwargs)

    out_text = tok.decode(out_ids[0], skip_special_tokens=True)
    print("\n===== OCR OUTPUT =====")
    print(out_text.strip())
    print("======================\n")

if __name__ == "__main__":
    main()
