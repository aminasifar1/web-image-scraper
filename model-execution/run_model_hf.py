from __future__ import annotations

import argparse

from inference import EndpointHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPAI inference for one image.")
    parser.add_argument("--image", type=str, required=True, help="Local path, URL, or base64 image")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/fhome/aaasidar/spai-hf",
        help="Directory containing config and checkpoint",
    )
    args = parser.parse_args()

    handler = EndpointHandler(path=args.model_dir)
    result = handler({"inputs": args.image})
    print(result)


if __name__ == "__main__":
    main()