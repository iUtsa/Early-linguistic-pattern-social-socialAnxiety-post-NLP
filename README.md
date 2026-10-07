# Interpretable linguistic signals in anxiety-related online language

This project is being revised after a research-validity audit of [arXiv:2601.11758v1](https://arxiv.org/abs/2601.11758v1). The existing paper and saved results are historical evidence, not newly reproduced research results. The current work studies explicitly defined language/label proxies; it does not establish clinical diagnosis, clinical onset, or screening utility.

The audit found synthetic control histories, PHQ-8 rather than anxiety outcomes in clinical comparisons, clinical effect-size constants in a metric generator, and evaluation defects. Read the [publication-readiness review](research/PUBLICATION_REVIEW.md), [revised experiment protocol](research/EXPERIMENT_PROTOCOL.md), and [manuscript redesign](research/MANUSCRIPT_REFRAME.md) before using results. The original README and metric-generation source are preserved under [research/archive](research/archive/README.original.md) for traceability. Historical result files have not been overwritten.

The uploaded examples now support two completed, exploratory community-language pilots: March training, May validation and unseen June authors. The 13-feature macro-F1 is 0.533 against mentalhealth and 0.577 against depression; TF-IDF achieves 0.858 and 0.917. Length/activity and keyword checks are also complete. May-selected thresholds improve linguistic LR to 0.594/0.641; a nonlinear control reaches 0.613/0.659, while TF-IDF remains at 0.859/0.921. Additional reviewer checks reproduce frozen inference and test validation-selected thresholds, comparison transfer, substantial verbatim reuse and a nonlinear control. Read the [submission decision](research/SUBMISSION_DECISION.md), [data card](research/DATA_CARD.md), [new findings](research/UPLOADED_DATA_FINDINGS.md), [aggregate evidence](research/uploaded_pilot_results.json), and [revised manuscript draft](research/REVISED_PILOT_MANUSCRIPT.md). These are community-label results, not diagnostic performance or reproduction of the preprint.

The complete version-1 archive has now been verified. [Frozen July/August evaluation and 40 training resamples](research/TEMPORAL_EXTENSION_FINDINGS.md) strengthen the result beyond the initial examples; they remain within one source.

The replacement manuscript is available as [Markdown](research/REVISED_PILOT_MANUSCRIPT.md), [PDF](research/REVISED_PILOT_MANUSCRIPT.pdf) and editable [DOCX](research/REVISED_PILOT_MANUSCRIPT.docx). These are human-review drafts, with actual ethics details and author declarations still to be completed. Rebuild the exports with `python scripts/render_research_manuscript.py` when Pandoc and pdflatex are available.

## Development setup

Python 3.12 is the tested runtime. The cloud environment already contains `/workspace/nlp-venv` with CPU PyTorch and the pinned dependencies in `requirements.txt`. For another environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned file includes the verified CPU PyTorch index. Default linguistic-feature experiments do not download Hugging Face models or spaCy language models. Legacy embedding experiments require separately available pretrained assets.

## Integrity checks and software validation

```bash
# From the repository root, using the prepared cloud virtual environment:
source /workspace/nlp-venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=2
export MPLCONFIGDIR=/workspace/nlp-cache/matplotlib XDG_CACHE_HOME=/workspace/nlp-cache
python -m unittest discover -s tests -v
python scripts/audit_research_artifacts.py --output /workspace/artifact-audit.json
python -m src.research_benchmark --help
```

Tests exercise known failure conditions, train-only preprocessing, correctly aligned embeddings, measured effect sizes, paired uncertainty, and an end-to-end synthetic benchmark. Passing them validates software behavior; it does not validate the historical anxiety scores.

## Revised research workflow

The new runner compares majority, length-only, pronoun-only, without-sentiment, full 13-feature, and TF-IDF logistic regression baselines on the same observed users. It tunes only on development users, bundles preprocessing with models, records provenance and hashes, and saves paired uncertainty. Optional chronological prefix experiments use a common real-user cohort and are not labelled clinical early detection.

```bash
python -m src.research_benchmark \
  --csv /workspace/private-data/posts.csv \
  --provenance /workspace/private-data/provenance.json \
  --output /workspace/private-results/new-run \
  --bootstrap 2000 --seed 42
```

The example input paths do not exist in this repository. Complete the [provenance template](research/provenance.example.json) from source documentation and read the [protocol](research/EXPERIMENT_PROTOCOL.md). Raw examples and clinical label/split tables have been supplied privately. Exact original splits/predictions, stable original control identities and clinical text/feature measurements remain missing; Kaggle RMHD version 1 and its stated CC0 license are now verified, all 15 authored uploads now match downloaded upstream files by SHA-256. The author reports an institutional determination exists but cannot share its details here; the required ethics information will be added privately to the submission copy. The Hugging Face demo supplies an app and model. Its model is not the newly fitted pilot models.

Keep private datasets and fitted text artifacts outside the Git checkout; the runner refuses to overwrite runs or write them here. Do not generate substitute author histories for missing real participants. `compute_missing_metrics_final.py` is now a compatibility entrypoint to this runner and no longer emits hard-coded clinical findings.

## Repository layout

- `src/research_benchmark.py`: revised observed-author benchmark and CLI.
- `src/uploaded_pilot.py`, `src/pilot_sensitivity.py`: audited monthly cohort construction and fixed matching/keyword checks.
- `src/reviewer_checks.py`, `src/nonlinear_check.py`: explicitly post-hoc frozen-inference, operating-point, comparator-transfer, near-duplicate and nonlinear controls.
- `src/temporal_extension.py`, `src/training_resampling.py`: frozen July/August evaluation, historical-absence sensitivity and complete training-account resampling.
- `src/research_validation.py`: fail-fast identity, label and split checks.
- `src/research_stats.py`: participant-level measured effects and BH correction.
- `scripts/audit_research_artifacts.py`: arithmetic/consistency audit of historical aggregates.
- `scripts/audit_uploaded_clinical_labels.py`, `scripts/summarize_uploaded_pilot.py`: aggregate label provenance, saved-prediction verification and figure export.
- `tests/`: regression and synthetic integration checks.
- `research/`: candid publication assessment, protocol, source records and redesign.
- `src/run.py`, other legacy modules, `models/`, and `results/`: earlier pipeline/artifacts; read the audit before treating them as evidence. The legacy embedding pipeline differs from the paper's primary post-level model.

The public [Hugging Face Space](https://huggingface.co/spaces/nimbus1011/anxiety-detectionNLP/tree/main) has not been changed. No paper revision has been published and no journal submission has been made.
