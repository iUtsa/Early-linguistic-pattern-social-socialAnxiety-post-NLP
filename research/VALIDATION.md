# Validation evidence

Current-instance checks completed on 7 October 2026 using Python 3.12 and `/workspace/nlp-venv`. Software tests validate code. The two real-data pilots measure observed community affiliation, not anxiety diagnosis.

| Check | Outcome |
|---|---|
| `python -m unittest discover -s tests -v` | 23 tests passed in 32.764 seconds; includes prior integrity/integration tests plus upload deduplication/checksums, UTC boundaries, past-author exclusion before text filtering, no future relabeling, deterministic balanced matching exact-test underflow, validation-threshold tie rules and exhaustive-vs-prefix near-duplicate agreement |
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
| Source release/license | Kaggle RMHD v1 and stated CC0 verified; all 15 authored filenames/byte lengths match fully paginated 225-file listing; upstream checksum identity unverified |
| Uploaded clinical label audit | Completed: 189 nonoverlapping train/dev/test IDs; `anxiety_label` exactly PHQ-8 ≥ 10; one discrepancy against supplied original depression binary label; no clinical feature effects computed |
| New independent external study | Not run: the two contrasts share some authors and the same source; upstream checksum identity and suitable independent data remain outstanding |
| Standalone scientific figure | Aggregate-only baseline PNG/PDF generated and visually inspected; writable Matplotlib/font cache check passed |

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
