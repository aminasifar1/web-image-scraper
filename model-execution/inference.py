from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from spai.config import get_custom_config
from spai.data.data_finetune import build_transform
from spai.models import build_cls_model


class EndpointHandler:
    """Hugging Face Inference Endpoint handler for SPAI."""

    DEFAULT_MODEL_DIR = Path("/fhome/aaasidar/spai-hf")
    DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "spai" / "weights" / "spai.pth"
    DEFAULT_CONFIG = DEFAULT_MODEL_DIR / "configs" / "spai.yaml"

    def __init__(self, path: str = "") -> None:
        self.model_dir = Path(path) if path else self.DEFAULT_MODEL_DIR
        self.threshold = float(os.getenv("SPAI_THRESHOLD", "0.6"))

        cfg_path = self._resolve_config_path()
        self.config = get_custom_config(str(cfg_path))

        self.device = self._resolve_device()
        self.model = build_cls_model(self.config)
        checkpoint_path = self._resolve_checkpoint_path()
        state_dict = self._load_state_dict(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_transform(is_train=False, config=self.config)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        inputs = data.get("inputs", data.get("image", data))

        if isinstance(inputs, list):
            return [self._predict_one(item) for item in inputs]
        return self._predict_one(inputs)

    def _predict_one(self, raw_input: Any) -> dict[str, Any]:
        image = self._load_image(raw_input)
        image_np = np.array(image)
        image_tensor = self.transform(image=image_np)["image"]

        if self.config.MODEL.RESOLUTION_MODE == "arbitrary":
            model_input = [image_tensor.unsqueeze(0).to(self.device)]
            feature_batch_size = self.config.MODEL.FEATURE_EXTRACTION_BATCH
            with torch.no_grad():
                logits = self.model(model_input, feature_batch_size)
        else:
            model_input = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(model_input)

        score = float(torch.sigmoid(logits).flatten()[0].item())
        predicted_label = int(score >= self.threshold)

        return {
            "score": score,
            "predicted_label": predicted_label,
            "predicted_label_name": "ai-generated" if predicted_label == 1 else "real",
            "threshold": self.threshold,
        }

    def _resolve_config_path(self) -> Path:
        env_cfg = os.getenv("SPAI_CONFIG")
        if env_cfg:
            cfg_path = Path(env_cfg)
            if cfg_path.exists():
                return cfg_path
            raise FileNotFoundError(f"SPAI_CONFIG points to a missing file: {cfg_path}")

        candidates = [
            self.DEFAULT_CONFIG,
            self.model_dir / "configs" / "spai.yaml",
            self.model_dir / "spai.yaml",
            self.model_dir / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Could not locate model config. Expected one of: "
            "configs/spai.yaml, spai.yaml, config.yaml, or SPAI_CONFIG env var."
        )

    def _resolve_checkpoint_path(self) -> Path:
        env_ckpt = os.getenv("SPAI_CHECKPOINT")
        if env_ckpt:
            ckpt_path = Path(env_ckpt)
            if ckpt_path.exists():
                return ckpt_path
            if "::" in env_ckpt:
                repo_id, filename = env_ckpt.split("::", 1)
                downloaded = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
                return Path(downloaded)
            raise FileNotFoundError(f"SPAI_CHECKPOINT points to a missing file: {ckpt_path}")

        candidates = [
            self.DEFAULT_CHECKPOINT,
            self.model_dir / "spai.pth",
            self.model_dir / "pytorch_model.bin",
            self.model_dir / "weights" / "spai.pth",
            self.model_dir / "spai" / "weights" / "spai.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pth_files = sorted(self.model_dir.glob("*.pth"))
        if pth_files:
            return pth_files[0]

        raise FileNotFoundError(
            "Could not locate model checkpoint. Expected one of: "
            "spai.pth, pytorch_model.bin, weights/spai.pth, spai/weights/spai.pth, "
            "or SPAI_CHECKPOINT env var."
        )

    @staticmethod
    def _resolve_device() -> torch.device:
        force_cpu = os.getenv("SPAI_FORCE_CPU", "0") == "1"
        if (not force_cpu) and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]

        if isinstance(checkpoint, dict):
            tensor_values = all(isinstance(v, torch.Tensor) for v in checkpoint.values())
            if tensor_values:
                return checkpoint

        raise RuntimeError(
            "Unsupported checkpoint format. Expected a dict with key 'model' or a raw state_dict."
        )

    def _load_image(self, raw_input: Any) -> Image.Image:
        if isinstance(raw_input, Image.Image):
            return raw_input.convert("RGB")

        if isinstance(raw_input, bytes):
            return Image.open(io.BytesIO(raw_input)).convert("RGB")

        if isinstance(raw_input, dict):
            if "bytes" in raw_input:
                raw_bytes = raw_input["bytes"]
                if isinstance(raw_bytes, str):
                    raw_bytes = base64.b64decode(raw_bytes)
                return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            if "b64" in raw_input:
                return Image.open(io.BytesIO(base64.b64decode(raw_input["b64"]))).convert("RGB")
            if "url" in raw_input:
                return self._load_image_from_url(raw_input["url"])
            if "path" in raw_input:
                return Image.open(Path(raw_input["path"])).convert("RGB")

        if isinstance(raw_input, str):
            if raw_input.startswith("http://") or raw_input.startswith("https://"):
                return self._load_image_from_url(raw_input)

            if raw_input.startswith("data:image") and "," in raw_input:
                _, encoded = raw_input.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

            maybe_path = Path(raw_input)
            if maybe_path.exists():
                return Image.open(maybe_path).convert("RGB")

            try:
                decoded = base64.b64decode(raw_input, validate=True)
                return Image.open(io.BytesIO(decoded)).convert("RGB")
            except Exception as exc:
                raise ValueError(
                    "String input is neither a valid URL, file path, nor base64 image payload."
                ) from exc

        raise TypeError(
            "Unsupported input type. Use a URL/path/base64 string, bytes, PIL.Image, "
            "or dict with one of keys: bytes, b64, url, path."
        )

    @staticmethod
    def _load_image_from_url(url: str) -> Image.Image:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run SPAI inference for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Image path/URL/base64 input")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(EndpointHandler.DEFAULT_MODEL_DIR),
        help="Directory with config/checkpoint",
    )
    parser.add_argument(
        "--checkpoint-ref",
        type=str,
        default=None,
        help="Optional HF checkpoint ref in the form repo_id::path/in/repo",
    )
    args = parser.parse_args()

    if args.checkpoint_ref:
        os.environ["SPAI_CHECKPOINT"] = args.checkpoint_ref

    handler = EndpointHandler(path=args.model_dir)
    result = handler({"inputs": args.image})
    print(result)


if __name__ == "__main__":
    _main()
