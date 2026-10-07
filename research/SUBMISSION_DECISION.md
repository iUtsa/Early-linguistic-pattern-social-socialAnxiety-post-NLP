# Evidence-based submission decision

Assessment date: 7 October 2026. This document separates problems in the original preprint from the contribution and remaining requirements of the replacement study. Journal quartiles are scope/ranking information, not thresholds of scientific validity.

## Decision and study route

The original **early social-anxiety detection** manuscript is not supported for submission. Community labels, synthetic control histories and a depression-scale clinical comparison do not measure that outcome. Changing a journal tier cannot repair those claims.

The replacement is an **exploratory applied study of comparator dependence in community-language classification**. Its defensible question is: when anxiety-community accounts are compared with nearby mental-health communities, does a small sentiment/pronoun/style representation provide discrimination and transfer comparable to lexical content? This is a different paper, with measured new account-level outcomes and no clinical or onset claim.

The completed experiments provide an assessable empirical contribution. They do not establish that the general idea is new or guarantee acceptance. I would pursue an appropriately scoped applied Q4 venue first, and consider Q3 only if its editors see value in the controlled case study. The available novelty evidence does not support prioritizing Q2. A substantive independent replication would strengthen that case considerably; external validation is **not a universal requirement** for every community-language journal paper. A narrowly bounded single-source exploratory study can be submitted with transparent limitations once its actual reporting and governance requirements are complete.

## What the closest work already establishes

This is a targeted reading of directly relevant studies, not an exhaustive systematic review. Different studies' scores are not an experimental ranking.

| Work read | What it already contributes | Increment the replacement can claim |
|---|---|---|
| [Yates et al. 2017](https://aclanthology.org/D17-1322/) and [Cohan et al. 2018](https://aclanthology.org/C18-1126/) | Observed-person splits, self-disclosed diagnoses, matched controls and exclusion of overt mental-health content | Applying audited accounts to this release and outcome; neither observed-author splitting nor interpretable language features are new methods |
| [Harrigian, Aguirre & Dredze 2020](https://aclanthology.org/2020.findings-emnlp.337/) | Five depression datasets, cross-platform and temporal transfer, lexical/LIWC/topic/embedding features, and confounding analysis. LIWC-only models still lose transfer performance; thematic and temporal artifacts are established concerns | A smaller anxiety-community case with two adjacent negative communities, source-validation operating points, same-target comparator transfer, fixed covariate restrictions and term perturbations. This is an extension of established concerns, not the discovery that mental-health models can fail to generalize |
| [Harrigian et al. 2021](https://aclanthology.org/2021.clpsych-1.2/) and [Harrigian & Dredze 2022](https://aclanthology.org/2022.clpsych-1.6/) | Data heterogeneity/access limits and changing validity of self-disclosures | Exact release/cohort traceability and an explicitly community-based estimand; no diagnostic-label improvement is claimed |
| [Mitrović et al. 2024](https://aclanthology.org/2024.clpsych-1.12/) | Anxiety/panic analysis of 1,930 Quora/Reddit posts, linguistic/emotion analysis and transformer classifiers | Different comparison communities, later unseen accounts, paired baselines and comparator-transfer checks; anxiety-language classification and lexical/emotion analysis alone are already covered |
| [Shani & Stade 2025](https://aclanthology.org/2025.clpsych-1.6/) | Explicit critique of forum membership as psychiatric measurement and recommendations for valid, dimensional, transdiagnostic measures | A concrete, reproducible community-language case illustrating the predictive distinction; no claim to have empirically validated anxiety measurement |
| [Marker et al. 2026](https://aclanthology.org/2026.clpsych-1.14/) | Person-level document representations and robustness on psychometric datasets | A deliberately smaller representation comparison for a different label; no competitive clinical-state-of-the-art claim |

The main desk-rejection risk is **incremental contribution**, after correcting validity. A method-oriented venue can reasonably find the routine models insufficiently novel. The answer is a focused applied robustness contribution and suitable scope, not more unsupported clinical claims or a larger list of algorithms.

## Findings that survive reviewer objections

- Original fixed-threshold LR results: macro-F1 0.533/0.577 versus TF-IDF 0.858/0.917. Paired bootstrap intervals support a large predictive difference on the same June accounts.
- **Operating point matters.** May-selected thresholds improve 13-feature LR to **0.594/0.641**, versus TF-IDF **0.859/0.921**. Remaining paired lexical advantages are **0.265 [0.253–0.276]** and **0.280 [0.270–0.289]**. A conclusion that these features contain no useful signal would be false. Their ROC-AUC is 0.631/0.701; the conclusion is relative predictive usefulness for this outcome.
- **Performance depends on the fitted comparison.** On the same target accounts after source train/validation overlap exclusions, transferred models perform worse than native models. Source-validation-threshold macro-F1 differences are -0.044 [-0.056–-0.031] and -0.061 [-0.073–-0.050] for linguistic LR; TF-IDF differences are -0.058 [-0.067–-0.050] and -0.043 [-0.049–-0.036]. ROC-AUC also declines. Training cohorts and hyperparameters change with the comparison, so this does not isolate the negative community as a causal factor. It supports dependence on the fitted comparison. It does not show lexical models always transfer less well, clinical specificity or independent external replication.
- **Substantial verbatim reuse does not explain these primary scores.** The exact five-word-shingle Jaccard >=0.8 audit identifies one/two June posts and one/two affected authors. Their exclusion changes primary macro-F1 by less than 0.0002. This conclusion concerns the stated reuse definition, not all semantic paraphrases.
- Balanced length/activity restrictions retain the large lexical advantage; their different prevalence and population prevent a causal interpretation of score reductions.
- Frozen keyword perturbations reduce lexical performance, but an incomplete term list and perturbation shift prevent a claim that all remaining signal is topic-independent or that keywords cause the original preprint's score.
- Frozen-model inference independently reproduces every retained primary 13-feature/TF-IDF probability to 1e-12. This is stronger traceability than merely checking saved aggregate arithmetic.

The nonlinear-feature control improves validation-threshold macro-F1 to 0.613/0.659, exceeding linguistic LR by 0.019 [0.008–0.030] and 0.018 [0.009–0.026], while TF-IDF still exceeds it by 0.246 [0.233–0.258] and 0.262 [0.252–0.273]. Thus a linear-model explanation only partly accounts for the deficit. Complete settings and outcomes are reported in the manuscript and [aggregate evidence](nonlinear_check_results.json). It tests the linear-model explanation without changing the primary experiment or treating a post-hoc analysis as preregistered.

## Requirements versus useful strengthening

| Item | Actual status | Consequence |
|---|---|---|
| Outcome and inference unit | Corrected to community affiliation of observed accounts; original clinical/onset claims removed | Resolves the original construct mismatch for the replacement's stated question |
| Empirical result traceability | Private inputs/models/predictions retained; public aggregate hashes and frozen protocols; arithmetic and inference verified | Supports the new reported computations, not original-study reproduction |
| Source/release/license | Identified Kaggle RMHD v1; publisher states CC0; all 15 authored filenames and byte lengths correspond to release listing | Source identity and stated terms substantially resolved. Checksum identity/completeness remain qualified because archive delivery is blocked by current network |
| Applicable institutional determination | Not supplied for this secondary study | Human authors must establish and report the applicable determination with their institution/journal; original-study approval is not a substitute. No approval number or exemption can be generated by code |
| Contribution/scope | Closest work already shows proxy/domain/lexicon limitations | A modest applied case study is defensible; a broad methodological or clinical novelty claim is not |
| Independent source, longer periods and repeated-training uncertainty | Not completed | Valuable strengthening, especially for Q2 and broad generalization claims; transparently optional if the paper is limited to the completed case study |
| Language, bots and undocumented multi-account behavior | English-oriented features and explicit placeholder checks; broader audits absent | Bound the population and conclusions; do not claim fully verified people or multilingual/general-population applicability |
| Public raw-data/model release | Deliberately absent; source URL and reproducible code supplied | Individual sensitive material is unnecessary for a useful public code/aggregate release; journal availability statement must accurately describe access |

## Human-author steps before submission

1. Confirm the applicable secondary-analysis institutional determination and complete the provided [research-use statement worksheet](RESEARCH_USE_STATEMENT.md). Determine whether any additional institutional/data-use condition applies to this release.
2. Confirm uploaded-file identity against the listed upstream files when archive access works, or describe the study accurately as analysis of supplied files with matching source paths/byte lengths. Do not claim checksums were verified against Kaggle when they were not.
3. Add the actual authors, affiliations, contributions, funding/conflicts and target venue. Verify every statement, table and citation in the replacement manuscript and follow its AI-assistance disclosure policy. No messages to editors or dataset owners have been sent.

These are concrete remaining requirements, not a demand to reproduce every possible robustness experiment. The prepared manuscript can be assessed as a bounded applied paper; calling it accepted, guaranteed Q4, clinical evidence or fully submission-ready would exceed what has been established.
