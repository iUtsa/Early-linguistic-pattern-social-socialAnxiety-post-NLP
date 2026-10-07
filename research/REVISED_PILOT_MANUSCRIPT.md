# Comparator dependence of linguistic and lexical models for anxiety-community language on Reddit

Research manuscript draft, 7 October 2026. Completed exploratory experiments on author-supplied files corresponding to RMHD version 1. The publisher-stated CC0 license is verified; upstream checksum identity, the applicable secondary-analysis institutional determination and human authorship/submission declarations remain outstanding. This draft replaces the unsupported clinical/early-detection framing; it does not claim reproduction of arXiv:2601.11758v1.

## Abstract

Community-derived mental-health labels can conflate writing context with the psychological construct a classifier is intended to measure. We evaluate the comparator dependence of 13 sentiment, pronoun and style features versus lexical content in Reddit anxiety-community classification. Author-supplied March, May and June 2022 exports correspond to RMHD version 1. Models train in March, tune in May and evaluate previously unobserved accounts in June. Anxiety–mentalhealth and Anxiety–depression comparisons contain 8,717 and 9,622 test accounts. At a fixed threshold of 0.5, 13-feature logistic regression achieves macro-F1 0.533 and 0.577, versus 0.858 and 0.917 for TF-IDF logistic regression. Validation-selected thresholds improve linguistic macro-F1 to 0.594 and 0.641, while lexical advantages remain 0.265 [95% account-bootstrap interval 0.253–0.276] and 0.280 [0.270–0.289]. A nonlinear 13-feature control reaches 0.613 and 0.659. Cross-comparator application reduces both models' discrimination on identical target cohorts after source-exposure exclusions. Balanced length/activity restrictions retain the lexical advantage; frozen disorder/community-term replacement reduces lexical macro-F1 to 0.698 and 0.807. Excluding substantial cross-split verbatim reuse changes primary macro-F1 by less than 0.0002. Linguistic features contain useful community signal, but neither linearity nor operating point fully explains their deficit relative to lexical content. These exploratory, single-source findings concern community affiliation, not anxiety diagnosis or onset. We release reproducible code, aggregate evidence and protocols; applicable institutional determination remains to be documented.

## 1. Introduction

Social-media text can provide observations of mental-health discussion at scale, but label validity constrains the conclusions available from predictive experiments. Dataset access, uncertain labels and sampling heterogeneity limit the field [2,3]. Psychological measurement requires distinguishing diagnosed disorders, self-reported symptoms, scale scores and behavioral proxies [5]. An author posting in an anxiety-related forum is participating in a particular social context; this alone does not establish an anxiety disorder, and posting elsewhere does not establish its absence.

Low-dimensional sentiment and pronoun features are inexpensive and readily inspected. Their interpretation is nevertheless ambiguous: they may reflect autobiographical genre, topic or help-seeking conventions shared across mental-health communities. Prior anxiety/panic research already combines lexical and emotion analysis [4], and multi-dataset depression research has evaluated lexical and lexicon representations under temporal and domain shift [8]. Neither interpretable features, transfer failure nor anxiety-community classification is novel by itself. The relevant question is how these signals behave when comparison communities discuss other mental-health concerns and evaluation moves to unseen accounts in a later month.

This exploratory study asks whether a fixed 13-feature representation distinguishes anxiety-community affiliation beyond simple length/pronoun baselines, how it compares with lexical content, and whether those results persist under fixed length/activity restrictions and keyword perturbations. The incremental contribution is an audited RMHD case study combining adjacent-community comparisons with identical-target transfer, operating-point and representation checks. We study two related contrasts rather than treating general-interest posts as healthy-person controls. We do not infer diagnosis, clinical screening utility, psychological mechanism or pre-onset risk.

## 2. Data and cohort construction

The data are CSV examples supplied by the study author, who attributes the main collection to Rani et al. [1]. That study describes 1,494,019 original posts from January 2019–August 2022 and monthly subreddit-specific exports; its raw Part A schema agrees with these examples. Its 800-post annotated Part B concerns four proposed root-cause categories and does not provide anxiety diagnoses. The study’s Kaggle link resolves to RMHD version 1 (dataset ID 3687885; modified 1 September 2023), whose publisher states CC0. All 15 supplied authored filenames and byte lengths match its fully paginated 225-file listing. This corroborates release correspondence without establishing upstream checksum identity or a complete sampling frame. Content hashes of the actual uploaded inputs are recorded. Observed authored exports cover Anxiety, mentalhealth, depression, lonely and SuicideWatch communities; only the first three enter the two fixed comparisons. General-interest exports without authors are excluded from account-level analyses.

We skip byte-identical files and quarantine noncanonical community cells, placeholder/invalid author identifiers and invalid numeric metadata. Reddit author identifiers are case-normalized. These are observed accounts, not verified independent people. The exports contain no original platform post IDs. We use numeric Unix timestamps in UTC because the timezone-free string field differs systematically from UTC and has no documented timezone. Observed coverage is not assumed complete merely because filenames name a month.

Retained posts require nonremoved bodies containing at least ten words after deterministic URL/markdown/unicode/whitespace cleaning. Cleaned titles and bodies are combined. Exact duplicate records are removed. Within a model window, normalized text shared by multiple accounts is quarantined and repeated text within an account is reduced to its earliest occurrence. Later windows remove text already retained in earlier model windows. Training construction does not use future community labels or future text to discard training records. A subsequent audit tests substantial verbatim reuse using an exact five-word-shingle similarity definition; semantic paraphrases remain a limitation.

For each binary comparison, an author observed in both selected communities within the assigned month is excluded using metadata-valid records before body filtering. Positive labels denote exclusive observed Anxiety affiliation in that month; negative labels denote the selected comparison community. This restriction removes ambiguous membership from the chosen estimand and selects an atypical subset of forum participants. It does not assign a psychiatric state.

March 2022 supplies training data, May supplies validation and June supplies test data. Validation and test accounts cannot occur in either selected community in supplied earlier records, including text-quality-excluded records and boundary-day records outside the three model windows. Activity outside the uploads remains unknown.

| Contrast | Train authors/posts | Validation authors/posts | Test authors/posts | Positive test authors |
|---|---:|---:|---:|---:|
| Anxiety–mentalhealth | 11,528 / 14,196 | 10,267 / 12,095 | 8,717 / 9,991 | 3,704 |
| Anxiety–depression | 13,049 / 15,967 | 11,473 / 13,504 | 9,622 / 11,058 | 3,631 |

No observed author or exact normalized text crosses these model splits. The contrasts share some Anxiety accounts; they are dependent analyses rather than independent replications. Detailed exclusion counts are recorded in the accompanying aggregate manifest.

## 3. Models and evaluation

The 13 linguistic features are VADER negative/neutral/positive/compound scores, TextBlob polarity/subjectivity, first-person singular pronoun rate/count, character count, word count, average word length, punctuation density and emoji count. These tools use English-oriented lexicons; language suitability has not been independently established for every retained post. Features are computed per post and averaged within account. TF-IDF uses concatenated cleaned account text, unigram/bigram features, minimum document frequency two and a maximum vocabulary of 50,000 features.

We compare six baselines: a majority classifier; logistic regression using three length features; two pronoun features; seven features excluding sentiment; all 13 features; and TF-IDF logistic regression. Scaling and vocabulary fitting use training accounts only. Each logistic regression selects C from {0.1, 1, 10} by validation macro-F1, with ties favoring smaller C. Models are retained without refitting on validation data. Seed is 42 and the decision threshold is 0.5. No test outcome selects hyperparameters, calibration or threshold. Learned probabilities are not clinical risks.

The primary metric is macro-F1. We also retain positive-class F1, precision/recall, accuracy, ROC-AUC, average precision, Brier score, log loss and confusion matrices. Paired contrasts use the same ordered test accounts. We compute 1,000 stratified account bootstrap replicates and percentile intervals conditional on the fitted models, observed splits and class counts. These intervals do not quantify uncertain psychiatric labels or sampling/population validity. Exact McNemar comparisons address differences in binary correctness. Additional contrasts are exploratory and unadjusted for multiplicity.

Two secondary checks were fixed before primary test metrics were inspected. First, a balanced June subset matches fixed bins of mean cleaned post length (0–49, 50–99, 100–149, 150–249, 250–499, 500–999, 1000+ words) and activity (1, 2, 3+ posts), retaining the smaller class count in each stratum. SHA-256 ordering based on account identifier and seed selects accounts without using predictions. Continuous residual imbalance is reported. The models and thresholds remain frozen, so this evaluates a changed target population at 50% prevalence.

Second, the frozen TF-IDF model evaluates June text after neutral-token replacement of 22 prespecified disorder/community terms. This finite-list perturbation does not remove all topical language and does not retrain the model. A separate 13-feature sensitivity uses the original 16-term list. Different masks prevent interpreting their score changes as identical interventions.

Additional checks were designed after primary June outcomes were inspected and are explicitly post hoc. Retained-model inference first reproduces saved primary probabilities to absolute tolerance 1e-12. Each retained LR model selects a threshold by May macro-F1 on {0.05, 0.06, ..., 0.95}, with ties favoring proximity to 0.5 and then the smaller threshold. C, preprocessing and weights remain unchanged.

To test a linear-model explanation, histogram gradient boosting uses the same 13 feature means, max_iter=200, l2_regularization=1, early_stopping=False and seed=42. Four max-leaf/learning-rate configurations ({15,31} × {0.05,0.1}) are selected by May macro-F1 at 0.5, followed by the same May threshold selection. Nothing is fit or selected using June labels.

Cross-comparator application evaluates the other comparison's retained LR models on the same target June cohort as native models. Accounts seen in source training/validation and accounts with exact source-training/validation text are excluded. Source models keep their source-May thresholds. This changes training cohorts and hyperparameters alongside the comparison community, and cannot isolate a causal effect of the negative community alone. Five-word-shingle set Jaccard >=0.8 is audited between June and March/May using lossless prefix candidate filtering and exact set verification. Authors with detected reuse are excluded in a sensitivity check; this definition does not cover arbitrary paraphrases.

## 4. Results

| Model | Anxiety–mentalhealth macro-F1 [95% interval] | Anxiety–depression macro-F1 [95% interval] |
|---|---:|---:|
| Majority | 0.365 [0.365, 0.365] | 0.384 [0.384, 0.384] |
| Length only | 0.466 [0.456, 0.476] | 0.493 [0.483, 0.503] |
| Pronouns only | 0.447 [0.439, 0.456] | 0.466 [0.458, 0.475] |
| Without sentiment | 0.480 [0.470, 0.490] | 0.521 [0.511, 0.531] |
| All 13 features | 0.533 [0.523, 0.544] | 0.577 [0.567, 0.588] |
| TF-IDF | 0.858 [0.850, 0.866] | 0.917 [0.911, 0.922] |

The 13-feature model outperforms the simpler linguistic baselines but trails TF-IDF by 0.324 [0.312, 0.336] and 0.340 [0.328, 0.351] macro-F1. Its ROC-AUC is 0.631 and 0.701, versus 0.931 and 0.973 for TF-IDF. Positive F1 is 0.356 and 0.393, versus 0.831 and 0.894. Majority bootstrap intervals collapse because stratification fixes the class counts and the classifier predicts a constant class.

The restricted subsets retain 7,248 and 7,260 accounts. Mean-length standardized differences decrease from -0.320 to -0.019 and -0.167 to -0.014; activity differences decrease from 0.094 to 0.024 and 0.069 to 0.013. On these subsets, 13-feature macro-F1 is 0.497 [0.486, 0.508] and 0.546 [0.535, 0.557]; TF-IDF is 0.848 [0.840, 0.857] and 0.909 [0.902, 0.916]. This persistent lexical advantage is not explained solely by the coarse measured length/activity differences. The subset design does not isolate a causal effect of removing these covariates.

Frozen TF-IDF keyword replacement reduces macro-F1 to 0.698 and 0.807, paired differences -0.159 [-0.169, -0.151] and -0.110 [-0.117, -0.103]. Remaining discrimination may still depend on unmasked topic or community-language cues. Under its different original keyword mask, the 13-feature model reaches 0.482 and 0.590, showing that perturbation effects depend on the comparison context.

The post-hoc validation-selected thresholds are 0.44/0.37 for linguistic LR and 0.42/0.37 for TF-IDF. The resulting June macro-F1 values are 0.594/0.641 and 0.859/0.921. Paired lexical advantages remain 0.265 [0.253, 0.276] and 0.280 [0.270, 0.289]. The linguistic positive recall increases from 0.260/0.293 to 0.532/0.642. Thus a fixed 0.5 threshold understates the representation's signal, but does not explain the ranking.

Gradient boosting selects 15 leaves and learning rate 0.1 in both comparisons. Its macro-F1 is 0.594/0.634 at 0.5 and 0.613/0.659 at May-selected thresholds 0.45/0.42. Improvements over threshold-selected linguistic LR are 0.019 [0.008, 0.030] and 0.018 [0.009, 0.026]. TF-IDF still exceeds nonlinear linguistic features by 0.246 [0.233, 0.258] and 0.262 [0.252, 0.273]. This tests one nonlinear control, not all attainable performance with these features.

| Target June comparison (source for transfer) | Eligible accounts | Linguistic native → transferred macro-F1 | TF-IDF native → transferred macro-F1 |
|---|---:|---:|---:|
| Anxiety–mentalhealth (Anxiety–depression) | 8,551 | 0.593 → 0.549 | 0.860 → 0.801 |
| Anxiety–depression (Anxiety–mentalhealth) | 9,497 | 0.641 → 0.580 | 0.921 → 0.878 |

![Operating-point and nonlinear controls](figures/reviewer_controls.png)

Figure: May-selected operating-point performance and paired lexical advantages; intervals condition on the fitted models and observed class counts.

Transfer excludes 166/125 source-exposed accounts. Paired transferred-minus-native macro-F1 differences are -0.044 [-0.056, -0.031] and -0.061 [-0.073, -0.050] for linguistic LR, and -0.058 [-0.067, -0.050] and -0.043 [-0.049, -0.036] for TF-IDF. Linguistic ROC-AUC declines from 0.630 to 0.574 and 0.701 to 0.615; lexical ROC-AUC declines from 0.931 to 0.905 and 0.973 to 0.948. This is within-collection comparator transfer, not external clinical validation. The two model families do not have a uniformly ordered transfer penalty.

The reuse audit identifies one/two June posts belonging to one/two accounts. Removing these accounts changes fixed-threshold primary macro-F1 by less than 0.0002 for either model. Substantial verbatim reuse under this definition therefore does not explain the primary score gap.

Only 205 and 234 June accounts have at least three eligible posts; most have one. No early-history experiment was conducted. The original preprint's post-level F1 and these account-level results use different populations and recipes and are not a matched before/after comparison.

## 5. Discussion

The small linguistic representation contains useful community-associated information, with stronger results under validation-selected thresholds and a nonlinear model. Lexical content remains substantially more useful for this particular outcome. The transfer results demonstrate dependence on the fitted comparison; the study does not prove that lexical models always transfer worse than linguistic models. The result qualifies a strong interpretation of sentiment/pronoun/style signals as anxiety-specific markers: these measures have limited ability to distinguish nearby mental-health discussion contexts in the supplied examples. Their coefficients do not establish psychological mechanism, and correlated/compositional sentiment variables prevent simple coefficient-magnitude rankings from establishing dominance.

High TF-IDF performance is consistent with separable community-language distributions; it supplies no clinical evidence. The keyword perturbation shows reliance on a finite set of disorder/community terms, while residual performance leaves broader topical dependence unresolved. These findings are compatible with measurement-validity concerns [5] and prior evidence of selection/longitudinal limitations [3,7]. They do not establish that all historical performance was caused by source confounding: that causal attribution would require controlled reconstruction of the same original experiment.

The chief limitations are unverified upstream checksum identity and sampling coverage, exclusive-affiliation selection, unknown activity outside the uploads, account rather than patient independence, potential bots/multiple accounts, unverified language suitability, absent original post IDs, unresolved semantic paraphrases and dependent comparisons from one collection source. Post-hoc analyses use already inspected June outcomes and support exploratory interpretation. Fixed-model bootstrap intervals omit training-set and label uncertainty. Coarse matching changes prevalence and population, and keyword replacement can introduce distribution shift. The narrow time separation is not external generalization.

An additional authorized collection/source or compatible observed-user dataset would strengthen generalization claims, with an untouched holdout for a newly fixed design. It is not needed to define the completed bounded case study, but broad claims about anxiety-language mechanisms or deployment are not supported here. SMHD offers a relevant self-reported-diagnosis/matched-control design [2], subject to current access and use conditions; self-report still differs from clinician assessment. A clinical anxiety study would require a validated anxiety outcome and a separate appropriate design. The supplied DAIC-WOZ PHQ-8 labels measure depression [6] and cannot be renamed anxiety.

## 6. Ethics, availability and research status

The author requested private local analysis of uploaded CSVs. The source study reports Victoria University Human Research Ethics Committee approval HRE23-005, dated 29 May 2023 [1]. This approval belongs to that study; applicability and an appropriate determination for this secondary analysis have not been supplied. The upstream publisher states CC0 for RMHD version 1. That resolves its stated dataset license without establishing a current-project institutional determination. Human author/institution and disclosure declarations remain to be completed before submission. No current-project approval, exemption or consent basis is inferred. Raw post texts, observed usernames, participant-ID labels and fitted vocabularies remain outside the Git checkout. Aggregate results, code, protocols and synthetic regression tests are included in the GitHub research release. Source acquisition and file-correspondence details are in the accompanying data card and manifests. The human authors must verify their data-use basis and applicable target-journal disclosure requirements, including substantial AI assistance with auditing, code and drafting.

These results are exploratory. No new clinical effects were computed, no synthetic control histories were treated as people, and no diagnosis/onset conclusion is supported. The paper offers a modest applied robustness contribution, positioned against established proxy/domain-validity work; it makes no first-method or clinical-state-of-the-art claim. Independent data would strengthen the study, while institutional and human-author declarations remain actual submission requirements.

## References

1. Rani S, Ahmed K, Subramani S. From Posts to Knowledge: Annotating a Pandemic-Era Reddit Dataset to Navigate Mental Health Narratives. *Applied Sciences*. 2024;14(4):1547. [doi:10.3390/app14041547](https://doi.org/10.3390/app14041547). Methods, dataset organization, ethics and availability sections reviewed; Kaggle version-1 metadata and stated license verified, upstream file checksum identity unresolved.
2. Cohan A, Desmet B, Yates A, Soldaini L, MacAvaney S, Goharian N. SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions. COLING 2018. [ACL C18-1126](https://aclanthology.org/C18-1126/).
3. Harrigian K, Aguirre C, Dredze M. On the State of Social Media Data for Mental Health Research. CLPsych 2021. [ACL 2021.clpsych-1.2](https://aclanthology.org/2021.clpsych-1.2/).
4. Mitrović S, Lithgow-Serrano OW, Schillaci C. Comparing panic and anxiety on a dataset collected from social media. CLPsych 2024. [ACL 2024.clpsych-1.12](https://aclanthology.org/2024.clpsych-1.12/).
5. Shani C, Stade EC. Measuring Mental Health Variables in Computational Research: Toward Validated, Dimensional, and Transdiagnostic Approaches. CLPsych 2025. [ACL 2025.clpsych-1.6](https://aclanthology.org/2025.clpsych-1.6/).
6. Kroenke K et al. The PHQ-8 as a measure of current depression in the general population. *Journal of Affective Disorders*. 2009. [doi:10.1016/j.jad.2008.06.026](https://doi.org/10.1016/j.jad.2008.06.026). Bibliographic metadata checked; full scale-validation article not reviewed here.
7. Harrigian K, Dredze M. Then and Now: Quantifying the Longitudinal Validity of Self-Disclosed Depression Diagnoses. CLPsych 2022. [ACL 2022.clpsych-1.6](https://aclanthology.org/2022.clpsych-1.6/).

8. Harrigian K, Aguirre C, Dredze M. Do Models of Mental Health Based on Social Media Data Generalize? Findings of EMNLP 2020;3774–3788. [ACL 2020.findings-emnlp.337](https://aclanthology.org/2020.findings-emnlp.337/). Methods, results, limitations and feature/hyperparameter appendices reviewed.
