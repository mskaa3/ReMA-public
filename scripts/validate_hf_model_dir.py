"""Check a downloaded HF export for missing files before starting GPU workers."""

import argparse
import json
from pathlib import Path


def validate_model_dir(directory):
    directory = Path(directory).resolve()

    def require_file(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing, empty, or invalid model file: {name} in {directory}")
        return path

    config = json.loads(require_file("config.json").read_text())
    if not config.get("model_type"):
        raise ValueError(f"config.json has no model_type in {directory}")
    json.loads(require_file("tokenizer_config.json").read_text())
    if (directory / "tokenizer.json").exists():
        require_file("tokenizer.json")
    elif (directory / "tokenizer.model").exists():
        require_file("tokenizer.model")
    else:
        require_file("vocab.json")
        require_file("merges.txt")

    for base in ("model.safetensors", "pytorch_model.bin"):
        index = directory / f"{base}.index.json"
        if index.exists():
            weight_map = json.loads(require_file(index.name).read_text()).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Empty or invalid weight_map in {index}")
            for name in set(weight_map.values()):
                require_file(name)
            return
        if (directory / base).exists():
            require_file(base)
            return
    raise ValueError(f"No complete Hugging Face model weights in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    validate_model_dir(args.directory)
    print(f"Hugging Face model files present: {args.directory}", flush=True)


if __name__ == "__main__":
    main()
