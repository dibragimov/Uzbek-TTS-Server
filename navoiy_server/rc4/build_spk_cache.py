#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cosyvoice-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cosy_root = Path(a.cosyvoice_root)
    sys.path.insert(0, str(cosy_root))
    sys.path.insert(0, str(cosy_root / "third_party/Matcha-TTS"))

    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    model_dir = Path(a.model_dir)
    with open(model_dir / "cosyvoice2.yaml", "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")},
        )

    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        str(model_dir / "campplus.onnx"),
        str(model_dir / "speech_tokenizer_v2.onnx"),
        "",
        configs["allowed_special"],
    )

    print("Extracting fixed speaker speech tokens...", flush=True)
    speech_token, _ = frontend._extract_speech_token(a.reference)
    print("Extracting fixed speaker speech features...", flush=True)
    speech_feat, _ = frontend._extract_speech_feat(a.reference)

    token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
    speech_feat = speech_feat[:, : 2 * token_len].contiguous()
    speech_token = speech_token[:, :token_len].contiguous()

    print("Extracting fixed speaker embedding...", flush=True)
    embedding = frontend._extract_spk_embedding(a.reference)

    info = {
        "001": {
            "prompt_text": a.instruction,
            "speech_token": speech_token.cpu(),
            "speech_feat": speech_feat.cpu(),
            "embedding": embedding.cpu(),
        }
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(info, out)
    print(
        f"Saved {out}: speech_token={tuple(speech_token.shape)} "
        f"speech_feat={tuple(speech_feat.shape)} embedding={tuple(embedding.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
