"""The only reward selection point; unsupported settings fail before rollout."""
import math
import json
import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit

from .base import validate_result
from ..graphprm_v2.rewards import validate_failure_positive_scale

SETTINGS = ("gnn_prm", "llm_coach", "handcrafted", "frozen_prm", "finetuned_prm", "gnn_only")
_CACHE = {}


def coach_backend():
    return os.environ.get("REMA_COACH_BACKEND", "api" if os.environ.get("REMA_COACH_BASE_URL") else "local_vllm")


def local_coach_identity():
    value = json.loads(os.environ.get("REMA_COACH_LOCAL_IDENTITY", "{}"))
    if (not isinstance(value, dict) or not value.get("revision")
            or value.get("model") != os.environ.get("REMA_COACH_MODEL")):
        raise ValueError("Local coach is not ready. Launch through hierarchical-rema-train.sh; "
                         "the supervisor starts vLLM before training.")
    return value


def reward_setting(args):
    # Old embedded callers remain compatible. CLI/launchers always set a provider.
    return getattr(args, "reward_setting", None) or ("gnn_prm" if getattr(args, "gfam_reward_model_pkl", "") else None)


def validate_reward_setting(args):
    setting = reward_setting(args)
    if setting in ("frozen_prm", "finetuned_prm", "gnn_only"):
        raise NotImplementedError(f"Reward setting {setting!r} is registered but not implemented")
    if setting is None:
        return
    if setting not in SETTINGS:
        raise ValueError(f"Unknown reward setting {setting!r}; choose from {SETTINGS}")
    if getattr(args, "final_answer_correctness_reward_only", False):
        raise ValueError("Legacy correctness-only switch cannot be combined with a reward provider")
    penalty = float(getattr(args, "gfam_bad_class_penalty", 1.))
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Bad-class penalty must be finite and nonnegative")
    validate_failure_positive_scale(os.environ.get("GFAM_VERIFIED_FAILURE_POSITIVE_SCALE", ".25"))
    if setting == "gnn_prm" and not getattr(args, "gfam_reward_model_pkl", ""):
        raise ValueError("gnn_prm requires --gfam-reward-model-pkl")
    if setting == "llm_coach":
        backend = coach_backend()
        if backend not in ("api", "local_vllm"):
            raise ValueError("REMA_COACH_BACKEND must be local_vllm or api")
        url = urlsplit(os.environ.get("REMA_COACH_BASE_URL", ""))
        if not os.environ.get("REMA_COACH_MODEL", "").strip() or url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("llm_coach requires REMA_COACH_MODEL and REMA_COACH_BASE_URL (OpenAI-compatible /v1 endpoint)")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Coach URL must not contain credentials, query or fragment")
        if backend == "local_vllm":
            if url.scheme != "http" or url.hostname != "127.0.0.1":
                raise ValueError("Managed local coach must use its loopback endpoint")
            local_coach_identity()
        for name, default, maximum in (("REMA_COACH_WORKERS", 4, 32), ("REMA_COACH_RETRIES", 2, 8),
                                       ("REMA_COACH_MAX_TOKENS", 4096, 65536)):
            value = int(os.environ.get(name, default))
            if value < (0 if name.endswith("RETRIES") else 1) or value > maximum:
                raise ValueError(f"{name} is outside its supported range")
        timeout = float(os.environ.get("REMA_COACH_TIMEOUT", 120))
        if not math.isfinite(timeout) or not 0 < timeout <= 1800:
            raise ValueError("REMA_COACH_TIMEOUT must be finite and in (0, 1800]")


class GraphPRMProvider:
    source = "gnn_prm"
    supports_verified_final_correctness = True

    def __init__(self, scorer):
        self.scorer = scorer
        self.rollout_batch_size = getattr(scorer, "rollout_batch_size", 1)

    def score_rollouts(self, requests):
        batch = getattr(self.scorer, "score_rollouts", None)
        results = batch(requests) if batch else [self.scorer.score_rollout(**r) for r in requests]
        if len(results) != len(requests):
            raise ValueError("Graph+PRM returned misaligned results")
        return [validate_result({**result, "source": self.source}, request)
                for result, request in zip(results, requests)]

    def score_rollout(self, **request):
        return self.score_rollouts([request])[0]


def build_reward_provider(args):
    validate_reward_setting(args)
    setting = reward_setting(args)
    if setting is None:
        return None
    if setting == "gnn_prm":
        from ..gfam_reward import load_cached_reward_scorer
        scorer = load_cached_reward_scorer(
            checkpoint_path=args.gfam_reward_model_pkl, device=args.gfam_reward_model_device,
            encoder_backend=args.gfam_reward_model_encoder_backend,
            encoder_model=args.gfam_reward_model_encoder_model,
            prm_device=getattr(args, "gfam_prm_device", "cpu"),
            bad_class_penalty=getattr(args, "gfam_bad_class_penalty", 1.0),
            prm_torch_dtype=getattr(args, "gfam_prm_torch_dtype", "auto"),
            prm_max_length=getattr(args, "gfam_prm_max_length", 0),
            feature_cache_dir=getattr(args, "gfam_feature_cache_dir", None),
            encoder_device=getattr(args, "gfam_encoder_device", None))
        if not getattr(scorer, "supports_verified_final_correctness", False):
            raise ValueError("gnn_prm requires a binary checkpoint with verified-outcome support")
        return GraphPRMProvider(scorer)
    options = dict(bad_class_penalty=float(getattr(args, "gfam_bad_class_penalty", 1.)),
                   failure_scale=validate_failure_positive_scale(
                       os.environ.get("GFAM_VERIFIED_FAILURE_POSITIVE_SCALE", ".25")))
    if setting == "handcrafted":
        from .handcrafted import HandcraftedRewardProvider
        return HandcraftedRewardProvider(**options)
    from .coach import CoachRewardProvider
    options.update(model=os.environ["REMA_COACH_MODEL"], base_url=os.environ["REMA_COACH_BASE_URL"],
                   backend=coach_backend(),
                   local_identity=json.dumps(local_coach_identity(), sort_keys=True) if coach_backend() == "local_vllm" else "",
                   api_key_env=os.environ.get("REMA_COACH_API_KEY_ENV", "REMA_COACH_API_KEY"),
                   cache_dir=os.environ.get("REMA_COACH_CACHE_DIR") or str(Path(args.output_dir) / "coach_cache"),
                   workers=int(os.environ.get("REMA_COACH_WORKERS", 4)),
                   timeout=float(os.environ.get("REMA_COACH_TIMEOUT", 120)),
                   retries=int(os.environ.get("REMA_COACH_RETRIES", 2)),
                   max_tokens=int(os.environ.get("REMA_COACH_MAX_TOKENS", 4096)))
    key = tuple(sorted(options.items()))
    if key not in _CACHE:
        _CACHE[key] = CoachRewardProvider(**options)
    return _CACHE[key]


def reward_manifest(args):
    setting = reward_setting(args)
    result = {"reward_setting": setting, "reward_contract": "label_aligned_compiled_v1",
              "bad_class_penalty": getattr(args, "gfam_bad_class_penalty", 1.),
              "verified_failure_positive_scale": float(os.environ.get("GFAM_VERIFIED_FAILURE_POSITIVE_SCALE", ".25")),
              "ground_truth_in_provider_input": False,
              "saturated_failure_filter": os.environ.get("GFAM_SKIP_SATURATED_FAILURES", "true"),
              "saturated_failure_threshold": os.environ.get("GFAM_SATURATED_FAILURE_THRESHOLD", ".95")}
    if setting == "gnn_prm":
        result["checkpoint"] = args.gfam_reward_model_pkl
        checkpoint = Path(args.gfam_reward_model_pkl).expanduser()
        if checkpoint.is_file():
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result["checkpoint_sha256"] = digest.hexdigest()
    if setting == "handcrafted":
        from .handcrafted import VERSION
        result["heuristic_version"] = VERSION
    if setting == "llm_coach":
        from .coach import PROMPT_VERSION, COACH_PROMPT
        result.update(coach_model=os.environ["REMA_COACH_MODEL"],
                      coach_backend=coach_backend(),
                      coach_endpoint="managed_loopback" if coach_backend() == "local_vllm" else os.environ["REMA_COACH_BASE_URL"],
                      coach_prompt_version=PROMPT_VERSION,
                      coach_prompt_sha256=hashlib.sha256(COACH_PROMPT.encode()).hexdigest(),
                      coach_temperature=0,
                      coach_workers=int(os.environ.get("REMA_COACH_WORKERS", 4)),
                      coach_timeout=float(os.environ.get("REMA_COACH_TIMEOUT", 120)),
                      coach_retries=int(os.environ.get("REMA_COACH_RETRIES", 2)),
                      coach_max_tokens=int(os.environ.get("REMA_COACH_MAX_TOKENS", 4096)))
        if coach_backend() == "local_vllm":
            result["coach_local_identity"] = local_coach_identity()
    return result


def write_reward_manifest(args, output_dir):
    """Never mix different reward experiments in a single artifact directory."""
    manifest = reward_manifest(args)
    path = Path(output_dir) / "reward_setting.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output directory has a different reward configuration; use a new run directory")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[hierarchical-rema] reward_setting={manifest['reward_setting']} manifest={path}", flush=True)
