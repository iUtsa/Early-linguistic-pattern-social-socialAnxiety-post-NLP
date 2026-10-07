# Manuscript redesign

Update, 7 October 2026: the first real-data community pilots and fixed sensitivity analyses have completed. The later [submission decision](SUBMISSION_DECISION.md) and completed replacement manuscript supersede speculative scope and unsupported templates. Read [uploaded-data findings](UPLOADED_DATA_FINDINGS.md) and the [working manuscript draft](REVISED_PILOT_MANUSCRIPT.md). This document remains the broader redesign plan; its abstract template and future experiments are not completed results.

Working document; no revised empirical results have been produced. Preserve arXiv v1 as a public historical version and disclose substantive corrections in a future revision. This file does not update arXiv or submit to a journal.

## Proposed title

**Interpretable linguistic signals in anxiety-related Reddit language: sensitivity to topic, author, and time**

If new data supply self-reported diagnostic labels with real matched controls, use **Interpretable language features for self-reported anxiety: a controlled robustness study**. Neither title should imply clinician-confirmed diagnosis.

## Replacement contribution statement

The contribution is a reproducible evaluation of linguistic and lexical prediction under adjacent-community comparisons, activity/length restrictions, term perturbations and temporal evaluation. Logistic regression and a small feature set enable inspection and low-cost replication, but are not methodological novelty. The completed experiments do not causally reconstruct or explain the original preprint's score; they support the bounded replacement study. See the [completed manuscript](REVISED_PILOT_MANUSCRIPT.md), [later-period findings](TEMPORAL_EXTENSION_FINDINGS.md) and [submission assessment](SUBMISSION_DECISION.md).

## Abstract scaffold

Do not submit this scaffold or fill it with the old score. Each bracket must be replaced from verified revised-run evidence.

“Language from anxiety-related online communities is often treated as a proxy for anxiety status, although source community and writing genre can confound predictive performance. We investigate the robustness of 13 interpretable linguistic features on [documented dataset and label construct], using [observed-user sample sizes] and [frozen evaluation conditions]. We compare logistic regression with majority, lexical, length-only and pronoun-only baselines, and evaluate [completed source/genre, chronological and compatible external checks]. [Measured findings with paired differences and uncertainty.] These findings support [limited conclusion actually justified by the experiments] and do not establish diagnosis or clinical onset. We provide code and [permitted reproducibility artifacts], with documented data-access and ethics constraints.”

## Section-level rewrite

| Section | Required change |
|---|---|
| Introduction | Define community affiliation, self-reported symptoms, scale scores and clinician diagnosis separately. Explain the specific missing knowledge about robustness; remove claims that simple linguistic prediction is a new standard |
| Related work | Discuss SMHD, matched-control studies, anxiety/panic comparisons, measurement validity, temporal validity and current author-level embedding work. Compare outcome definitions and evaluation units before reporting others' scores |
| Data | Give a cohort-flow diagram, exact source/version, permissions, temporal window, observed author counts, label/exclusion rules, control matching and cross-split duplicate checks. Describe known test-data reuse |
| Methods | State whether each target is a person or a post. Document train-only transforms, validation-only model selection, feature scales, paired comparisons and the bootstrap sampling unit |
| Results | Generate tables from saved run manifests/predictions. Include baseline deficits, confound controls, uncertainty and external failures. Keep exploratory PHQ-8 associations separate from anxiety evidence |
| Interpretation | Replace causal/self-focus-theory assertions with conditional associations. Discuss correlated features, selection, source-community signals and incompatible clinical outcomes |
| Limited-history analysis | Include only if BOTH groups have real observed histories. Use matched cohorts and elapsed-time summaries. Label it retrospective history-length sensitivity unless a defined future outcome is actually predicted |
| Limitations | Missing labels/identities, control selection, time drift, unknown diagnoses, external outcome mismatch, retrospective test reuse and unmeasured demographics |
| Ethics and availability | Report the institutional determination and actual data-use conditions; release appropriate code/derived evidence while protecting identities. Follow target journal's AI-assistance policy |
| Conclusion | State the result, intended scope and tested failure conditions. Remove claims of generalizable screening, clinical onset, or validated psychological mechanism unsupported by those measurements |

## Claim ledger for the current manuscript

| Existing claim | Revision rule |
|---|---|
| “89.34% F1” | Historical aggregate, arithmetically consistent with saved confusion matrix; do not present as newly reproduced. Retain only if attributable individual predictions and recipe recover it |
| “author-disjoint ensures no leakage” | Replace with exactly audited conditions; absence of stable control identities blocks unseen-person claims |
| “clinical anxiety validation” | Remove. PHQ-8 groups concern depression symptoms; clinically valid anxiety labels were not demonstrated |
| “Hedges' g = .78–.92” | Remove as empirical evidence until actual participant-level measurements reproduce the values |
| “75% generalization” | At most an explicitly defined direction-agreement statistic with disclosed exclusions. It is not classifier accuracy, clinical generalization, or 75% prediction validity |
| “three posts ≈ one week” | Remove unless observed elapsed-time distributions demonstrate it in the relevant cohort |
| “only 0.90pp loss proves early detection” | Remove; comparison mixes units/cohorts and synthetic controls. Replace with a matched real-user sensitivity analysis if possible |
| “1.6× larger coefficient validates self-focused attention” | Remove; compare standardized effects/stability descriptively and avoid mechanism or causal claims |
| “keyword independent” | Restrict to measured sensitivity under the documented finite term list and preprocessing. Topic/genre confounds still require direct controls |
| “screening score/confidence” in demo | Label as model probability of the declared dataset class, without unvalidated high/moderate/low clinical bands. Do not equate balanced-source scores with population risk |

## Suggested revised tables and figures

Tables: cohort provenance and exclusions; baseline metrics with paired contrasts; community/genre controls; chronological/external transfer; feature stability and multiplicity-adjusted participant associations only if actual measured data exist.

Figures: cohort flow; paired performance changes with uncertainty; performance by observed history length plus cohort retention; calibration curves on compatible held-out users; coefficient stability rather than a single “dominant psychological marker” ranking.

No new figures should reuse unverifiable clinical effect sizes or synthetic early-history conclusions. Existing figures remain historical artifacts pending regeneration from supported results.
