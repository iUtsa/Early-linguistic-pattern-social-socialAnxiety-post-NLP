# Uploaded-data pilot protocol

Fixed on 7 October 2026 before fitting these pilot models. This is an exploratory protocol, not a preregistration. The uploaded examples and historical paper have already been inspected. No model selection will use the June outcomes.

## Question and scope

Can the existing 13 linguistic features distinguish authors posting in r/Anxiety from authors posting in another mental-health community, after removing shared authors and evaluating later calendar months? Two fixed comparisons are r/Anxiety versus r/mentalhealth and r/Anxiety versus r/depression. Label 1 is Anxiety community membership in the assigned month; label 0 is the comparison community. Neither label establishes presence or absence of a disorder. The comparisons assess community-language separability and specificity limits, not social-anxiety diagnosis, onset or early clinical detection.

The uploads are examples supplied by the author. Upstream dataset URL/version, completeness, redistribution license, ethics determination and consent basis are unresolved. Local exploratory analysis was requested; this is not permission to publish raw posts or usernames. The two authorless r/datasets comment exports, employment and whatsbotheringyou files cannot supply author-level controls. Other authored communities are inventoried but not added after viewing model outcomes.

## Cohort construction

Implemented in `src/uploaded_pilot.py`. Preserve raw files outside Git. Verify their manifest SHA-256 values, and skip byte-identical files. Select only files with the observed author/post schema. Quarantine noncanonical community cells, invalid Reddit-author syntax or placeholder IDs, and invalid numeric timestamps/scores. Author case is normalized because Reddit usernames are case-insensitive. These observed identifiers are not independent verification of account ownership or the population sampling process.

Use numeric `created_utc` Unix seconds, not the filename or the timezone-free `timestamp` string. The initial March files have a consistent 11-hour offset between the two fields when their strings are interpreted as UTC. No conversion is inferred from an undocumented string timezone. Calendar windows are March 2022 (train), May 2022 (validation), June 2022 (test), in UTC. Boundary-day records from February/April are not model samples, but earlier observed accounts still count toward later exclusion.

Require a nonempty, nonremoved body with at least 10 whitespace-delimited words after the existing deterministic cleaning. Use cleaned title plus body for features. Report all exclusions. Do not equate a CSV row index or a derived content hash with an observed Reddit post ID: these authored exports have no original post IDs.

Within each assigned month, exclude any author observed in both comparison communities, using metadata-valid records before filtering short/removed bodies. This defines a selected exclusive-affiliation cohort. Do not infer labels from an author's later-month activity. Validation/test authors must not have appeared in either selected community in any supplied earlier record, including records excluded from modeling because of text quality. There is no assertion that their activity outside these uploads is known.

Deduplicate identical `(author, UTC time, community, normalized text)` records. Process model windows chronologically: within each window quarantine normalized text shared by multiple accounts, retain the earliest occurrence of a repeated text within one account, and remove later-window text already retained in an earlier model window. Training construction does not inspect later-window text to remove training rows. Exact deduplication does not resolve paraphrases, quotation or other near duplicates. Counts of affected records and author/time ties are reported; no prefix experiment is run in this pilot.

Keep observed class proportions. Do not resample the test set to balance classes. All features and predictions use one row per author; linguistic features are means over that author's eligible posts in the month, while TF-IDF uses concatenated text. Thus this pilot also does not reproduce the paper's original post-level recipe or histories.

## Analysis fixed before fitting

Run the six existing baselines: majority, length-only, pronoun-only, features without sentiment, all 13 features, and unigram/bigram TF-IDF logistic regression. Training-only scaling/vocabulary; C in {0.1, 1, 10}, selected using validation macro-F1; deterministic ties favor the smaller C. Threshold 0.5. No test tuning, balancing, calibration or train-plus-validation refit. Seed 42.

Primary estimand is June author-level macro-F1; the primary contrast is TF-IDF versus all 13 features. Report class prevalence, positive F1, ROC-AUC, average precision, Brier score, confusion matrix and all baselines even if unhelpful. Use 1,000 stratified author bootstrap replicates for pilot 95% percentile intervals and paired differences, conditional on these fitted models and observed class counts. These intervals do not capture label uncertainty or variation across training datasets. Two comparisons and secondary contrasts are exploratory; unadjusted p-values cannot establish a confirmatory family of discoveries.

Also apply the already-defined leak-term mask to June text under the frozen 13-feature model. This probes only the finite keyword list and can change sentiment/length. It is not a comprehensive topic control. Length matching, near-duplicate analysis, independently acquired cohorts, masked retraining, broader topic controls and diagnostic-label validation remain subsequent requirements.

## Interpretation and artifacts

This is a new, attributable exploratory analysis of the supplied examples. It cannot validate historical aggregate tables, restore missing control authors, or supply missing clinical participant measurements. Report incomplete-source sampling and exclusive-affiliation selection. A weak result or a stronger TF-IDF baseline is evidence that should change the paper's contribution.

Save prepared data, fitted vocabularies and models outside the checkout with restrictive local permissions. Commit only aggregate reports, code, protocol and synthetic software tests. Run-local prediction IDs are pseudonyms, not permission to redistribute individual records. A submission-ready paper still requires confirmed provenance/terms, an appropriate ethics statement, defensible novelty, additional controlled/generalization experiments and a complete manuscript rewritten around supported results.
