WORKING ACL SHORT-PAPER DRAFT

main.tex is the editable source. references.bib contains verified primary-source
references. acl.sty and acl_natbib.bst are unmodified copies of the user's
Downloads/acl-style-files-master template. Review mode anonymizes the authors.
No conference acceptance, completed planned experiment, or statistically
significant result is implied.

FRAMING
The paper leads with learning a useful division of reasoning: successful answers
can conceal bypass of intended worker computation. Task completion and capability
acquisition are different objectives. Guidance should enable worker computation
without replacing it, while full-question access can permit independent re-solving.
The problem formulation separates final correctness, desired functional cooperation,
and measurable eligibility. Specialisation is motivated by reusable worker skills,
not by assuming that more agents or stronger dependence is inherently better.
The method is named EAC3 (Eligibility-Aware C3). It is a role-scoped adaptation
of C3 that distinguishes factual success, known protocol failure, and unknown audits
when defining action scores and updates. Its conceptual distinction is attribution
versus eligibility: accurately crediting answer disclosure does not fix a mismatch
between final-answer reward and the intended division of work. Shared prefixes, repeated continuations,
and leave-one-out comparison are inherited from C3, not claimed as new. The
all-continuation gate changes the target with M; the modified learning rule does
not inherit an unbiasedness guarantee for raw-outcome C3. The appendix distinguishes
the worker update-masking rule from the evaluated decomposer reward rule.
No claim of being the first planner/worker architecture or first use of probing
is made. The empirical comparison is presented as experimental results, without
repeated pilot or matched-checkpoint labels. The training budget remains in the
reproducibility appendix; all reported values and their provenance are unchanged.
This framing does not imply convergence, statistical significance, completed
planned ablations, or evaluation beyond the documented audit.
MAST is cited to acknowledge existing work on multi-agent failure modes. The
paper does not claim to discover role separation problems or to prove that all
successful shortcuts prevent learning. Independent cooperation and worker-transfer
tests, rather than training-gate pass rates alone, are the central future evidence.

BUILD
  cd papers/eligibility-aware-reasoning
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The delivered PDF is also copied to:
  output/pdf/eligibility-aware-reasoning.pdf

PROBLEM FIGURE
Figure 1 contrasts planner disclosure, whole-question worker bypass, and a useful
division of reasoning for the same pen-count problem. All three return 98 and
receive raw outcome reward 1. The panels are constructed illustrations, not new
empirical evidence or claims about automatic gate decisions. Arrows highlight
selected result use; the caption explains the omitted instruction routing.

Editable vector artwork:
  figures/problem-overview.tex
It is included directly in main.tex with TikZ, so no raster asset is required.
To export the figure separately, run from this directory:
  latexmk -pdf -interaction=nonstopmode -halt-on-error problem-overview-standalone.tex
The standalone deliverable is:
  output/pdf/eac3-problem-overview.pdf
The source ZIP must retain the figures/ subdirectory for Overleaf compilation.
Page-count reduction is deferred while the problem presentation is developed.

ARCHITECTURE FIGURE
Figure 2 shows the execution path: question, decomposer, deterministic routing,
sequential worker stages, and the terminal answer. Input labels distinguish
non-terminal question access from the terminal stage's task-and-results context.
All worker stages share one model; trainable/frozen labels refer to the reported
decomposer-training phase. Optional training-only teacher context and EAC3 reward
computation are omitted to keep the architecture separate from the learning rule.

Editable vector artwork:
  figures/architecture.tex
Build its standalone export with:
  latexmk -pdf -interaction=nonstopmode -halt-on-error architecture-standalone.tex
The standalone deliverable is:
  output/pdf/eac3-architecture.pdf
Include architecture-standalone.tex and figures/architecture.tex in the source ZIP.

RESULTS
The main results are the step-30 128-trajectory leakage audits and the 51
identity-matched printed examples. They are NOT full-suite validation accuracy.
The runs change both eligibility handling and group handling. The probe model
also evolves with the decomposer. These confounds are explicit in the paper.

The original baseline log is no longer available locally. Its retained clean
copy supplies the baseline metrics. The October 11 treatment excerpt has a
complete final audit summary, but a truncated aggregate metric line. Therefore
the treatment's per-benchmark step-30 numbers are PENDING, not fabricated or
borrowed from another step. Treatment audit size 128 follows the configured cap;
the complete treatment metric record should confirm it before submission.

Evidence is in evidence/step30.json, the two extracted log records,
matched-questions.json, and qualitative-examples.json. Source hashes and paths
record provenance. These files contain local paths and run IDs; anonymize them
before releasing a review artifact. The collector does not rerun scoring.

REPRODUCE EXTRACTION
  python3 collect_evidence.py \
    --baseline /private/tmp/6061597-clean.txt \
    --eligibility '/Users/arkadiuszjanz/.codex/attachments/bf985db2-c3c9-4aa9-8d87-5f3155b397f9/Pasted text.txt'

PENDING EXPERIMENTS
1. Same-code ablation: raw C3, minimum-two-task constraint, eligibility masking,
   and EAC3's eligibility-conditioned reward; repeat seeds and match model checkpoints.
2. Frozen-probe and manual audits plus relevant-versus-irrelevant worker-result
   interventions. Report probe coverage and false positives/negatives.
3. Worker transfer and alternating training with matched total compute, including
   probes, teacher generation, and all continuations; strong single-agent baselines.

The new evaluation appendix specifies these comparisons and a proposed reusable
trace protocol separating correctness, audit validity, eligibility, independent
annotations, and realised training credit. This is not a completed dataset release
or an implemented new logging feature. Capability acquisition, accuracy recovery,
and applications beyond cooperative mathematics remain hypotheses.

Keep the draft's PENDING labels until measurements actually exist. Do not infer
causal cooperation, improved worker skills, or universal leakage reduction from
the training-gate pass rate. Main-paper length and final venue requirements must
be checked again after filling the planned experiments.
