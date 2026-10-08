# Validation evidence

Current-instance checks completed on 7 October 2026 using Python 3.12 and `/workspace/nlp-venv`. Software tests validate code. The two real-data pilots measure observed community affiliation, not anxiety diagnosis.

| Check | Outcome |
|---|---|
| `python -m unittest discover -s tests -v` | 27 tests passed in 35.636 seconds; includes prior integrity/integration tests plus upload deduplication/checksums, UTC boundaries, past-author exclusion before text filtering, no future relabeling, deterministic balanced matching exact-test underflow, validation-threshold tie rules and exhaustive-vs-prefix near-duplicate agreement |
| Revised benchmark CLI on an external synthetic fixture, `--ks 3 --bootstrap 100` | Completed with `status=synthetic_software_validation`; six baselines, paired full/prefix outputs, frozen keyword sensitivity, manifest and model artifacts retained in `/workspace/nlp-research-validation/run1` |
| Prepared environment's feature/training/evaluation/SHAP/plotting smoke after repairs | Passed on synthetic data; no Hugging Face download or real-study prediction was used |
| Pinned requirements installation and `python -m pip check` | Passed; no broken requirements |
| Revised CLI help, Python source parsing, `git diff --check` | Passed |
| Historical result audit | Main accuracy, precision, recall, F1 agree with saved confusion matrix; author totals, direction-agreement denominators and McNemar convention differences recorded in `audit_summary.json` |
| Original experiment reproduction | Not run: partial raw examples and clinical label manifests now supplied, but exact original processed data/splits, original held-out predictions and clinical text/features remain unavailable |
| Anxiety-specific clinical validation | Not run: compatible validated anxiety outcome and independent participant data absent |
| New time-forward observed-author community pilots | Completed both fixed comparisons with six baselines and 1,000-author intervals; 8,717/9,622 June authors; no shared observed authors or exact normalized texts across model splits |
| Fixed secondary length/activity and keyword checks | Completed two matched-cohort and two frozen lexical-mask analyses; matched June subsets contain 7,248/7,260 authors; no model retuning |
| Saved individual prediction/aggregate arithmetic | Verified confusion matrices and macro-F1 independently from saved predictions; all primary and secondary artifact checksums passed |
| Post-hoc reviewer checks | Completed frozen inference parity to 1e-12, May-only threshold selection, same-target comparator transfer and five-word-shingle reuse audit; no primary models changed |
| Nonlinear representation control | Four validation-selected histogram-boosting configurations fit March features only; June macro-F1 .613/.659 at May thresholds; saved probabilities and artifact hashes independently checked |
| Source release/license | Kaggle RMHD v1 and stated CC0 verified; complete archive and 225 extracted files verified; all 15 authored uploads are byte-identical by SHA-256 |
| Uploaded clinical label audit | Completed: 189 nonoverlapping train/dev/test IDs; `anxiety_label` exactly PHQ-8 ≥ 10; one discrepancy against supplied original depression binary label; no clinical feature effects computed |
| Frozen later-period extension | Completed four July/August cohorts with three retained models, May thresholds, historical-absence/reuse sensitivities and 1,000 paired account bootstraps; all saved prediction arithmetic/hashes verified |
| Training-account resampling | Completed all 20 seeds in each comparison; scaling/vocabulary/LR refitted on stratified March-account resamples, thresholds selected on May only; positive lexical advantage in all 40 runs |
| New independent external study | Not run: the two contrasts share some authors and the same source; suitable independent data remain outstanding |
| Standalone scientific figures | Aggregate-only baseline/reviewer/temporal PNG/PDF figures generated and visually inspected; writable Matplotlib/font cache check passed |
| Expanded novelty audit | 16 Crossref queries return 382 unique DOIs; complete ACL metadata snapshot has 131,647 records and 432 automated candidates; five arXiv metadata queries and targeted close-paper reading documented in `novelty_search_manifest.json`. Counts describe retrieval/screening, not full-text reading of every record or worldwide uniqueness |
| Elsevier journal-fit search | 30 journal-specific searches across 13 journals return 467 unique DOIs; 13 reproduced scopes/SJR profiles read, seven author-version queries and one matching temporal-modeling preprint examined. Direct ScienceDirect/Journal Finder/guides remain proxy-blocked; no current APC or official recommendation-engine result claimed. Scope/ranking records, source hashes and limits are in `elsevier_journal_search_manifest.json` |
| Manuscript after literature update | PDF/DOCX regenerated; source, figure, rendering-script and output hashes verified. Citations and local review links checked; updated title, figures and tables visually inspected. Experimental code and aggregate outcomes unchanged |

The 100-replicate software fixture supplies no empirical evidence. The exploratory upload pilots use their separately fixed 1,000-replicate protocols; the proposed future study protocol recommends 2,000. Bootstrap intervals condition on observed class prevalence and the fitted model. These are not intervals for label validity or clinical/population generalization.

The exact primary-run code was preserved privately and is now also released in `research/archive/pilot-primary-code`, verified against recorded source hashes, before improving numerical representation of very small exact p-values. The aggregate export preserves original artifact hashes and recomputes exact probabilities in log space from discordant counts; private original reports remain intact. No models were refitted for this numerical correction.

The public research release is traced in Git history. Historical models/results remain intact. No arXiv revision, journal submission or remote demo update occurred. Future machines and published snapshots have not been independently validated.

## Reproduce the additional checks with the retained private inputs

```bash
python -m src.reviewer_checks --private-root /workspace/research-private \
  --output /workspace/research-private/new-reviewer-run --bootstrap 1000
python -m src.nonlinear_check --private-root /workspace/research-private \
  --reviewer-output /workspace/research-private/new-reviewer-run \
  --output /workspace/research-private/new-nonlinear-run --bootstrap 1000
python -m scripts.summarize_reviewer_checks \
  --reviewer-directory /workspace/research-private/new-reviewer-run \
  --nonlinear-directory /workspace/research-private/new-nonlinear-run --destination research
```

The current completed private runs are `reviewer-checks-seed42` and `nonlinear-check-seed42`. The initial reviewer protocol is preserved under `research/protocol_snapshots` with the exact run hash; the nonlinear addition is dated and explicitly post hoc. These commands require the retained inputs and model files; they are not evidence that private data are available in a fresh clone.

## Reproduce the frozen extension

For a fresh machine, obtain version 1 from the Kaggle link in the data card, verify the archive SHA-256 there, and extract it into a private directory outside the checkout. Recreate the input manifests from the public source record, preserving its file ordering and recorded hashes:

```python
import json
from pathlib import Path

record = json.loads(Path('research/rmhd_source_manifest.json').read_text())
private = Path('/workspace/research-private/rmhd-v1')
files = private / 'files'  # extracted archive; not a directory in the Git release
release = [{'name': f['path'], 'path': str(files / f['path']),
            'size_bytes': f['bytes'], 'sha256': f['sha256']}
           for f in record['upstream_files']]
uploads = [{'name': f['upload_name'], 'path': str(files / f['source_path']),
            'sha256': f['upload_sha256']}
           for f in record['upload_correspondence']]
(private / 'manifest.json').write_text(json.dumps(release, indent=2))
(private / 'authored_inputs.json').write_text(json.dumps(uploads, indent=2))
```

For a fresh reproduction, use `authored_inputs.json` as the upload manifest; the current instance retains the original `upload-audit/uploads.json` below. First reconstruct the March/May/June cohorts and primary models using the exact archived primary code, the [pilot cohort policy](UPLOADED_PILOT_PROTOCOL.md) and the [retained-data commands](EXPERIMENT_PROTOCOL.md), then run the reviewer/nonlinear checks above. Organize those runs under the directory names expected by the reviewer/extension CLIs. The original primary provenance records predate source identification; the current source manifest/data card provide its verified release attribution. A new run records its own paths/hashes and must not be represented as the original run merely because its measured results agree.

```bash
python -m src.temporal_extension \
  --release-manifest /workspace/research-private/rmhd-v1/manifest.json \
  --upload-manifest /workspace/research-private/upload-audit/uploads.json \
  --private-root /workspace/research-private \
  --output /workspace/research-private/new-temporal-run --bootstrap 1000
python -m src.training_resampling --private-root /workspace/research-private \
  --output /workspace/research-private/new-training-resamples
python -m scripts.summarize_temporal_extension \
  --temporal-directory /workspace/research-private/new-temporal-run \
  --resampling-directory /workspace/research-private/new-training-resamples --destination research
```

The completed private outputs are `temporal-extension-seed42` and `training-resampling-seeds100-119`. Source data/history and individual predictions are deliberately absent from the Git release. Complete archive retrieval was verified in this instance; access can still depend on the execution environment's network policy.

## Render the manuscript draft

`python scripts/render_research_manuscript.py` uses the existing Pandoc/pdflatex installation to create the PDF and editable DOCX. Input and output hashes and renderer versions are recorded in `manuscript_export_manifest.json`. These are working drafts for human review, not submitted or accepted documents. Rendering adds no research results or authorship/ethics assertions.
