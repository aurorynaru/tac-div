"""Download the exact model files the Pixelle-Video workflows reference.

Uses the huggingface_hub Python API directly (the `hf` CLI is unusable with the
ComfyUI embedded Python, which lacks the `venv` module). Resumable: files that
already exist are skipped, and huggingface_hub continues partial downloads from
its cache. The HF token is read from the HF_TOKEN environment variable only.
"""
import os
import sys

from huggingface_hub import hf_hub_download

MODELS = r"C:\AI\ComfyUI\ComfyUI_windows_portable\ComfyUI\models"

# (repo_id, repo-relative filename, short label)
FILES = [
    ("black-forest-labs/flux-3-image-klein-comfy",
     "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors", "FLUX3 UNET"),
    ("black-forest-labs/flux-3-image-klein-comfy",
     "text_encoders/qwen3vl_4b_int8_convrot.safetensors", "FLUX3 text encoder"),
    ("black-forest-labs/flux-3-image-klein-comfy",
     "vae/flux2-vae.safetensors", "FLUX3 VAE"),
    ("QuantStack/Wan2.2-I2V-A14B-GGUF",
     "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", "Wan2.2 HighNoise GGUF"),
    ("QuantStack/Wan2.2-I2V-A14B-GGUF",
     "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", "Wan2.2 LowNoise GGUF"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "umt5-xxl fp8"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/vae/wan_2.1_vae.safetensors", "Wan 2.1 VAE"),
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Gated FLUX 3 files will fail.")
    print(f"Models dir: {MODELS}")
    failures = []
    for repo, rel, label in FILES:
        dst = os.path.join(MODELS, rel)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  (already present) {label}: {rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"  downloading {label}: {repo}/{rel} ...", flush=True)
        try:
            hf_hub_download(repo_id=repo, filename=rel, local_dir=MODELS)
            ok = os.path.exists(dst) and os.path.getsize(dst) > 0
            print(f"  {'OK' if ok else 'MISSING'} {label}")
            if not ok:
                failures.append(label)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {label}: {e}")
            failures.append(label)
    print("=== Summary ===")
    if failures:
        print("Failed/missing:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All 7 model files present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())