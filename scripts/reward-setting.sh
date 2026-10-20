#!/usr/bin/env bash
# Source before downloads/model loading. One immutable choice per training job.
export REWARD_SETTING=${REWARD_SETTING:-gnn_prm}
case "$REWARD_SETTING" in
    gnn_prm|llm_coach|handcrafted) ;;
    frozen_prm|finetuned_prm|gnn_only)
        python3 -c 'import os; raise NotImplementedError("Reward setting %r is not implemented" % os.environ["REWARD_SETTING"])'
        return 1 ;;
    *) echo "Unknown REWARD_SETTING=$REWARD_SETTING" >&2; return 1 ;;
esac
if [[ "$REWARD_SETTING" == "llm_coach" ]]; then
    export REMA_COACH_BACKEND=${REMA_COACH_BACKEND:-local_vllm}
    case "$REMA_COACH_BACKEND" in
        local_vllm)
            export REMA_COACH_MODEL=${REMA_COACH_MODEL:-Qwen/Qwen3-32B}
            export REMA_COACH_REVISION=${REMA_COACH_REVISION:-main}
            export REMA_COACH_MAX_MODEL_LEN=${REMA_COACH_MAX_MODEL_LEN:-8192}
            export REMA_COACH_MAX_NUM_SEQS=${REMA_COACH_MAX_NUM_SEQS:-1}
            export REMA_COACH_GPU_MEMORY_UTILIZATION=${REMA_COACH_GPU_MEMORY_UTILIZATION:-0.90}
            export REMA_COACH_STARTUP_TIMEOUT=${REMA_COACH_STARTUP_TIMEOUT:-1800}
            export REMA_COACH_SHUTDOWN_TIMEOUT=${REMA_COACH_SHUTDOWN_TIMEOUT:-30}
            export REMA_COACH_SEED=${REMA_COACH_SEED:-42}
            export REMA_COACH_WORKERS=${REMA_COACH_WORKERS:-1}
            export REMA_COACH_TIMEOUT=${REMA_COACH_TIMEOUT:-300}
            # Resolved locally by the supervisor, never reuse another service.
            unset REMA_COACH_BASE_URL REMA_COACH_LOCAL_IDENTITY REMA_COACH_RESOLVED_REVISION
            ;;
        api)
            : "${REMA_COACH_MODEL:?Set REMA_COACH_MODEL for llm_coach}"
            : "${REMA_COACH_BASE_URL:?Set REMA_COACH_BASE_URL to your OpenAI-compatible /v1 endpoint}"
            export REMA_COACH_MODEL REMA_COACH_BASE_URL
            export REMA_COACH_API_KEY_ENV=${REMA_COACH_API_KEY_ENV:-REMA_COACH_API_KEY}
            # Export the configured secret by NAME; never print or write its value.
            if [[ ! "$REMA_COACH_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
                echo "Invalid REMA_COACH_API_KEY_ENV" >&2; return 1
            fi
            export "$REMA_COACH_API_KEY_ENV"
            export REMA_COACH_WORKERS=${REMA_COACH_WORKERS:-4}
            export REMA_COACH_TIMEOUT=${REMA_COACH_TIMEOUT:-120}
            ;;
        *) echo "Unknown REMA_COACH_BACKEND=$REMA_COACH_BACKEND" >&2; return 1 ;;
    esac
    export REMA_COACH_RETRIES=${REMA_COACH_RETRIES:-2}
    export REMA_COACH_MAX_TOKENS=${REMA_COACH_MAX_TOKENS:-4096}
fi
