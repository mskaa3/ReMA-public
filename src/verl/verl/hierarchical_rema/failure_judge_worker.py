"""Standalone GPU worker: can run under a separate Gemma-compatible interpreter.

No imports from the training package: its pinned vLLM/Transformers remain untouched.
"""
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys

from failure_judge_contract import OUTPUT_SCHEMA, PROMPT, VERSION, digest, parse_annotation, validate_annotation


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_requests(documents, tokenizer, config, emit):
    requests = []
    for key, document in documents.items():
        messages = [{"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps(document, ensure_ascii=False)}]
        try:
            tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            count = len(tokens)
            if count + config["max_tokens"] > config["max_model_len"]:
                emit({"input_id": key, "status": "context_overflow", "input_tokens": count,
                      "error": "Complete input plus output budget exceeds context; no truncation"})
            else:
                requests.append((key, document, tokens))
        except Exception as exc:
            emit({"input_id": key, "status": "tokenization_error", "error": type(exc).__name__})
    return sorted(requests, key=lambda item: len(item[2]))


def judge_batches(requests, generate, batch_size, retries, emit):
    """One result per content ID; retries and split-on-error preserve alignment."""
    def batch(items, attempt=0):
        if not items:
            return
        try:
            outputs = generate([{"prompt_token_ids": item[2]} for item in items])
            if len(outputs) != len(items):
                raise ValueError("Generation count mismatch")
        except Exception as exc:
            if len(items) > 1:
                middle = len(items) // 2
                batch(items[:middle], attempt)
                batch(items[middle:], attempt)
            elif attempt < retries:
                batch(items, attempt + 1)
            else:
                emit({"input_id": items[0][0], "status": "generation_error",
                      "attempts": attempt + 1, "error": type(exc).__name__})
            return
        retry = []
        for (key, document, tokens), result in zip(items, outputs):
            try:
                if result["finish_reason"] != "stop":
                    raise ValueError("Incomplete output")
                annotation = parse_annotation(result["text"], document)
                emit({"input_id": key, "status": "ok", "annotation": annotation,
                      "input_tokens": len(tokens), "attempts": attempt + 1, "cache_hit": False})
            except (ValueError, TypeError, KeyError):
                if attempt < retries:
                    retry.append((key, document, tokens))
                else:
                    emit({"input_id": key, "status": "invalid_annotation", "attempts": attempt + 1,
                          "input_tokens": len(tokens), "error": "Incomplete JSON or invalid schema/evidence"})
        # Retry only failed items together, never duplicate successful judgments.
        batch(retry, attempt + 1)

    for start in range(0, len(requests), batch_size):
        batch(requests[start:start + batch_size])
        print(f"[failure-judge] judged={min(start + batch_size, len(requests))}/{len(requests)}", flush=True)


def run(job):
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    config = job["config"]
    model_config = AutoConfig.from_pretrained(config["model"], revision=config["revision"],
                                              trust_remote_code=False)
    revision = getattr(model_config, "_commit_hash", "")
    if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise ValueError("Judge model must resolve to an immutable HF Hub snapshot")
    identity = {"model": config["model"], "resolved_revision": revision,
                "versions": {name: metadata.version(name) for name in ("vllm", "transformers", "torch")},
                "engine": "vllm", "dtype": "bfloat16", "thinking": False,
                "temperature": 0, "top_p": 1, "tensor_parallel_size": 1,
                "seed": config["seed"], "prompt_sha256": digest(PROMPT), "schema_sha256": digest(OUTPUT_SCHEMA),
                "version": VERSION, "batch_size": config["batch_size"],
                "max_model_len": config["max_model_len"], "max_tokens": config["max_tokens"]}
    atomic_json(Path(job["identity_path"]), identity)
    cache = Path(job["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    keys = {key: digest({"input_id": key, "judge": identity}) for key in job["documents"]}
    with Path(job["results_path"]).open("a", buffering=1, encoding="utf-8") as handle:
        def emit(result):
            if result["status"] == "ok" and not result.get("cache_hit"):
                atomic_json(cache / (keys[result["input_id"]] + ".json"), result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

        pending = {}
        for key, document in job["documents"].items():
            try:
                result = json.loads((cache / (keys[key] + ".json")).read_text())
                validate_annotation(result["annotation"], document)
                if result["input_id"] != key or result["status"] != "ok":
                    raise ValueError("Invalid cache identity")
                emit({**result, "cache_hit": True})
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pending[key] = document
        if not pending:
            return
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision, trust_remote_code=False)
        requests = build_requests(pending, tokenizer, config, emit)
        if not requests:
            return
        llm = LLM(model=config["model"], revision=revision, tokenizer_revision=revision,
                  dtype="bfloat16", tensor_parallel_size=1, distributed_executor_backend="mp",
                  max_model_len=config["max_model_len"], max_num_seqs=config["batch_size"],
                  max_num_batched_tokens=config["max_model_len"],
                  gpu_memory_utilization=config["gpu_memory_utilization"],
                  enable_prefix_caching=True, enforce_eager=True, seed=config["seed"],
                  generation_config="vllm", trust_remote_code=False)
        params = SamplingParams(temperature=0, top_p=1, max_tokens=config["max_tokens"],
                                seed=config["seed"], structured_outputs=StructuredOutputsParams(json=OUTPUT_SCHEMA))

        def generate(prompts):
            outputs = llm.generate(prompts, params, use_tqdm=False)
            if [list(output.prompt_token_ids) for output in outputs] != [p["prompt_token_ids"] for p in prompts]:
                raise ValueError("vLLM request alignment mismatch")
            return [{"text": output.outputs[0].text, "finish_reason": output.outputs[0].finish_reason}
                    for output in outputs]
        judge_batches(requests, generate, config["batch_size"], config["retries"], emit)


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            run(json.load(handle))
    except Exception:
        print("[failure-judge] Startup/evaluation failed. Gemma requires a compatible vLLM/Transformers "
              "environment and HF model access. No package upgrades were attempted.", flush=True)
        raise
