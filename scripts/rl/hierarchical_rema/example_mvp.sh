#!/usr/bin/env bash

set -euo pipefail

PYTHONPATH=src/verl/verl python -m hierarchical_rema.demo \
  --backend mock \
  --mode joint \
  --num-decompositions 3 \
  --num-selections 2 \
  --soft-max-hops 3 \
  --hard-max-hops 5
