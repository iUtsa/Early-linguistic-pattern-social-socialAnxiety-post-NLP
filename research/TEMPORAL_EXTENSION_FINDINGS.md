# Later-period evaluation and training-sample sensitivity

7 October 2026. The [extension protocol](TEMPORAL_EXTENSION_PROTOCOL.md) was fixed after June results were inspected and before the new July/August evaluations. These are frozen retrospective later-period checks within one collection, not registered clinical validation.

## Source identity and coverage

The complete Kaggle RMHD version-1 archive was downloaded privately. All 225 files were hashed and matched the public file listing; all 15 authored uploads are byte-identical to their corresponding upstream files. Archive SHA-256 is `6078fe83304c266ca976973f8b1e553dc5d818c59810508151f9e6bc615bf9e4`. The earlier blocked-download and checksum-identity limitations are resolved.

There are 220 raw paths: 219 usable CSVs and one Apple Numbers file. Two byte-identical duplicate file pairs were skipped. Parsed distinct-file CSV rows total 1,834,871, with 1,813,026 metadata-valid rows under the documented policy. The Numbers file (`depmay21.numbers`) is not part of historical CSV coverage. Raw export row counts are not verified unique original-post counts; the source article reports 1,494,019 posts after English cleaning, so these definitions must not be conflated.

## Frozen later-period results

All weights, preprocessing, vocabulary and May-selected thresholds remain unchanged. July/August accounts exclude model/development/June exposure in supplied selected-community records and earlier extension-month activity, including text-quality-excluded records. Class labels remain exclusive observed community affiliation. Comparisons share some accounts and must not be pooled as independent replications.

| Comparison | Month | Accounts | Linguistic LR macro-F1 [95% CI] | Linguistic boosting | TF-IDF LR | Paired TF-IDF minus linguistic LR [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Anxiety–mentalhealth | 2022-07 | 9,074 | 0.591 [0.581, 0.601] | 0.604 | 0.854 | 0.263 [0.251, 0.275] |
| Anxiety–mentalhealth | 2022-08 | 9,016 | 0.596 [0.585, 0.605] | 0.610 | 0.860 | 0.264 [0.252, 0.275] |
| Anxiety–depression | 2022-07 | 9,520 | 0.640 [0.631, 0.650] | 0.647 | 0.926 | 0.286 [0.276, 0.296] |
| Anxiety–depression | 2022-08 | 8,859 | 0.646 [0.637, 0.656] | 0.650 | 0.922 | 0.275 [0.264, 0.286] |

![Later-period evaluation](figures/temporal_extension.png)

The lexical advantage remains large in all four later-period cohorts, including against the nonlinear linguistic control. The results support short-term persistence within this release. Distinct accounts, changing prevalence and eligibility selection prevent interpreting month-to-month score differences as a causal time effect. All four cohorts contain at least 100 accounts in each class.

## Stricter historical absence and substantial-reuse checks

The historical-absence sensitivity excludes any account with earlier metadata-valid selected-community activity in parseable full-release CSV records from 2019 onward. This means not observed in those records, not a verified new Reddit user or complete lifetime absence. The unparsed Numbers file and unknown activity elsewhere remain limitations.

| Comparison/month | Historically absent accounts | Linguistic LR | Boosting | TF-IDF | Lexical advantage [95% CI] |
|---|---:|---:|---:|---:|---:|
| mentalhealth/2022-07 | 7,629 | 0.588 | 0.603 | 0.857 | 0.269 [0.256, 0.281] |
| mentalhealth/2022-08 | 7,805 | 0.597 | 0.613 | 0.862 | 0.265 [0.252, 0.278] |
| depression/2022-07 | 7,786 | 0.640 | 0.648 | 0.928 | 0.288 [0.277, 0.300] |
| depression/2022-08 | 7,450 | 0.645 | 0.650 | 0.924 | 0.279 [0.267, 0.291] |

The five-word-shingle Jaccard >=0.8 audit against March/May texts detects zero affected accounts in three cohorts and one in the Anxiety–depression August cohort. Complete exclusion sensitivity metrics are retained. This covers substantial verbatim reuse, not arbitrary semantic paraphrases.

## Training-account resampling

Twenty stratified March-account bootstrap resamples per comparison (seeds 100–119) retain the primary LR C, refit scaling/vocabulary and weights, and select thresholds using May only. June evaluates every run; no best seed replaces the primary model. Values below are descriptive means, standard deviations and ranges across these runs, not population confidence intervals.

| Comparison | Linguistic LR mean ± SD | TF-IDF mean ± SD | Lexical advantage mean ± SD | Advantage minimum–maximum |
|---|---:|---:|---:|---:|
| Anxiety–mentalhealth | 0.593 ± 0.003 | 0.854 ± 0.002 | 0.262 ± 0.003 | 0.254–0.267 |
| Anxiety–depression | 0.639 ± 0.002 | 0.917 ± 0.001 | 0.278 ± 0.002 | 0.274–0.281 |

The lexical advantage is positive in all 40 runs and is large relative to training-resample variation. This does not address uncertain psychiatric labels, collection bias or an independent source.

## Submission assessment update

The replacement study now has verified release identity, four previously unevaluated later-period cohorts and an explicit training-sample sensitivity analysis. This improves its reproducibility and empirical depth. It remains an incremental applied study: prior literature already establishes proxy/domain limitations, and one source cannot establish external clinical generalization. An appropriately scoped Q4 submission is a defensible route after actual ethics/author declarations; Q3 remains scope-dependent. These results do not establish acceptance odds or a Q2-level contribution.

See [submission decision](SUBMISSION_DECISION.md), [revised manuscript](REVISED_PILOT_MANUSCRIPT.md), [complete temporal aggregates](temporal_extension_results.json) and [all resampled-training results](training_resampling_results.json). Saved probabilities, confusion matrices, macro-F1 arithmetic, input hashes and artifact checksums were verified independently before aggregate export. Raw posts, account identifiers and individual predictions remain private.
