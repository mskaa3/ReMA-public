RESERVED REVIEW PACKAGE: NOT READY FOR INFERENCE

This is the original reserved sample, not a new draw. No model responses were read.
The JSON files are pending authoring/review templates, NOT completed annotations.
Draft 1-4 coherent tasks and local references for every question. A distinct human
checker must verify them; record real provenance. Include or exclude every row,
with a substantive reason for each exclusion. Never exclude by model performance.
Do not mark checks true automatically. Do not invent a checker or human review.
Do not place the final/local reference answer in a compute-task instruction.
See docs/eac3_worker_specialization_study.md for the schema and checklist.

After all batches have been checked:
python3 -B scripts/eac3_worker_study_pack.py assemble --package experiments/worker-specialization-300-v1/review --output experiments/worker-specialization-300-v1/review/completed-review.json
