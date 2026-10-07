# Revised experiment protocol

Status: proposed prospective plan for a revised study, not a preregistration and not an empirical result. The old study test set has already been inspected. Resolve dataset/label provenance before freezing this plan.

Update: the supplied examples have their own [fixed pilot protocol](UPLOADED_PILOT_PROTOCOL.md), [secondary protocol](UPLOADED_PILOT_SECONDARY_PROTOCOL.md), [completed results](UPLOADED_DATA_FINDINGS.md) and working manuscript. They do not complete all future-study requirements below. June pilot outcomes have now been inspected; any new confirmatory design needs a fresh holdout.

## Required inputs

- A locally authorized CSV with `author`, `text`, binary `label`, and `split` (`train`, `val` or `validation`, `test`). Use stable observed pseudonymous author identifiers in BOTH classes. One label per author is required for this user-level estimand; mixed source labels require a scientific resolution outside the runner.
- For prefix experiments, observed timestamps in `created_utc`, `timestamp`, or `created_at`. Numeric timestamps are Unix seconds; string timestamps must parse in UTC. Ties require a documented ordering rule and are currently rejected. `posts_seen` alone does not establish chronology.
- Completed metadata based on `research/provenance.example.json`. `author_id_origin` is `observed_stable`, `synthetic_grouping`, or `generated_fixture`. Real data with synthetic groups cannot enter an author-level benchmark. Permission metadata is a record of the user's basis, not automated permission verification.
- Independent train/development/test users with both classes in every split. Normalize/deduplicate before assigning partitions under a documented policy. The runner fails on shared normalized text across partitions rather than silently removing it.

## Commands

Run from the repository root in the prepared environment:

```bash
source /workspace/nlp-venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=2
python -m unittest discover -s tests -v
python scripts/audit_research_artifacts.py --output /workspace/research-artifact-audit.json

# User-level full-history benchmark. Output directory must be new and outside Git.
python -m src.research_benchmark \
  --csv /workspace/private-data/posts.csv \
  --provenance /workspace/private-data/provenance.json \
  --output /workspace/private-results/full-seed42 \
  --bootstrap 2000 --seed 42

# Separate, secondary analysis: SAME users at k=3,5,10 and full history.
python -m src.research_benchmark \
  --csv /workspace/private-data/posts.csv \
  --provenance /workspace/private-data/provenance.json \
  --output /workspace/private-results/prefix-seed42 \
  --ks 3 5 10 --bootstrap 2000 --seed 42
```

These data paths are examples, not files included in the repository. With missing study data the commands should fail, not produce paper tables. The retired `compute_missing_metrics_final.py` invokes the same CLI and requires the same arguments.

## Fixed choices in the implemented runner

One sample per observed author. Linguistic features are the mean of per-post features; TF-IDF uses concatenated cleaned user text. Majority, length-only, pronoun-only, without-sentiment, 13-feature and TF-IDF LR baselines share users. Feature extraction is fixed; scaling and vocabulary fitting use training authors only. LR C candidates are 0.1, 1 and 10, selected by validation macro-F1 with deterministic grid-order ties; no test tuning or train+validation refit. Threshold is fixed at 0.5, and probabilities remain uncalibrated.

Primary proposed comparison: macro-F1, 13-feature versus TF-IDF LR. Additional metrics: positive F1, precision, recall, accuracy, ROC-AUC, average precision, Brier score, log loss, confusion matrix and test prevalence. Bootstrap samples authors within class, with percentile intervals conditional on observed class counts, fitted model, and split. This is not uncertainty over label validity, data collection, training sets, or clinical deployment prevalence.

Paired contrast outputs use B minus A on the same ordered test authors. Exact McNemar examines differences in binary correctness, not differences in F1. The bootstrap estimates paired metric differences directly. Additional comparisons have per-comparison intervals and unadjusted exploratory p-values; correct prespecified families before confirmatory interpretation. Calibration plots and grouped calibration fitting are not yet implemented in this runner.

Extremely small exact probabilities use a finite `log10_pvalue`, with `pvalue=null` and `pvalue_underflow=true` when the ordinary floating-point value underflows. This is a numerical representation issue, not evidence of an exact zero probability.

Prefix analyses keep users with at least max(k) posts, then evaluate the same users for every horizon and full history. This conditions on future activity and may select atypical, high-activity users; report the retention count per class and do not equate this retrospective cohort with a prospective population. Timestamps give first-observed history, not diagnosis onset. All users in either class must be real.

Frozen-model keyword masking replaces the predefined leak-term list with the neutral token `term` before feature computation. This preserves an interpretable perturbation but can change sentiment and token counts. It does not establish independence from all disorder-related language; masked retraining, explicit-term removal, genre matching and community exclusion remain separate planned checks.

## Artifact and privacy requirements

Each run writes a JSON manifest with input/code/package hashes, Git revision and dirty state, metadata, split counts, model selection, uncertainty and artifact checksums. Model files bundle scaler/vectorizer and classifier. Predictions use run-local numeric author IDs; raw usernames/text are not included in prediction CSVs. Classifier/vectorizer artifacts can reveal training vocabulary; keep all outputs private until a data-use review permits release. Do not publish a mapping from numeric IDs to people.

The runner intentionally refuses to overwrite a run or write private artifacts inside the checkout. It does not reconstruct the original study's post-level recipe, resolve missing real identities, verify the truth of declared provenance, automatically create new partitions, or implement matched controls, near-duplicate detection, held-out-community splits, external prediction, transformer comparisons, calibrated thresholds, or prospective diagnosis outcomes. These are scientifically important next experiments, not completed features.

## Decision rules

If topic/genre-matched performance approaches the majority/length baseline, present source confounding as the result. If TF-IDF wins, quantify the interpretability/compute tradeoff rather than hiding it. If external performance collapses, restrict generalization claims. If original predictions cannot be recovered, keep historical tables out of the revised confirmatory results. A valid result is one the design supports, including an inconvenient or null result.

## Reproduce the retained upload pilot

The commands below use retained private files in this cloud instance. They are not paths to publicly distributed data. New output directories are required; the existing completed runs must not be overwritten. Confirm source/use conditions before transferring these inputs to another environment.

```bash
source /workspace/nlp-venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/workspace/nlp-cache/matplotlib XDG_CACHE_HOME=/workspace/nlp-cache

# Repeat separately for comparison=mentalhealth and comparison=depression.
python -m src.uploaded_pilot \
  --manifest /workspace/research-private/upload-audit/uploads.json \
  --comparison mentalhealth --output /workspace/private-results/new-mentalhealth-data
python -m src.research_benchmark \
  --csv /workspace/private-results/new-mentalhealth-data/posts.csv \
  --provenance /workspace/private-results/new-mentalhealth-data/provenance.json \
  --output /workspace/private-results/new-mentalhealth-seed42 --bootstrap 1000 --seed 42
python -m src.pilot_sensitivity \
  --data-dir /workspace/private-results/new-mentalhealth-data \
  --run-dir /workspace/private-results/new-mentalhealth-seed42 \
  --output /workspace/private-results/new-mentalhealth-sensitivity

# Audit the additional clinical label uploads and verify/export completed pilot artifacts.
python scripts/audit_uploaded_clinical_labels.py \
  --manifest /workspace/research-private/upload-audit/additional_uploads.json \
  --output /workspace/research-private/upload-audit/clinical_label_audit_reproducible.json
python scripts/summarize_uploaded_pilot.py \
  --private-root /workspace/research-private --output research/uploaded_pilot_results.json
```

The summarizer reads the retained `pilot-mentalhealth-*` and `pilot-depression-*` run directories. Reproduced runs under different names require a correspondingly organized private root; the example rerun paths above do not replace retained artifacts. Numerical p-value formatting was improved after the primary runs; original code/hash snapshots remain under `/workspace/research-private/pilot-primary-code`. Model-fitting rules and reported prediction metrics were not changed by that formatting repair.
