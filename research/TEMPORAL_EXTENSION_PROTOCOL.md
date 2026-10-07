# Frozen later-period extension

Fixed 7 October 2026 before evaluating the newly accessible July/August files. Primary March–June results, reviewer checks and their limitations have already been inspected. This is a prospective extension of an exploratory study, not a registered confirmatory clinical experiment.

## Question

Does the measured lexical advantage over the 13-feature linguistic representation persist in later, previously unevaluated months under frozen models and operating points? Additional periods are within RMHD version 1 and are **not an independent source**.

## Inputs and cohort policy

Obtain the identified complete version-1 archive privately and verify all supplied authored files against extracted source SHA-256 values. Inventory the full release without displaying individual cells. Do not convert annotations into diagnoses. Report archive/file hashes and parsing coverage; unsupported file types are not silently treated as observed records.

Evaluate UTC July and August 2022 separately for Anxiety–mentalhealth and Anxiety–depression. Retain the existing title/body cleaning, >=10-word body requirement, observed-author policy, finite numeric timestamp/score checks, exclusive monthly selected-community affiliation and exact record/text deduplication. Remove a test author if any retained input post exactly matches a source training/validation post. Do not retune or refit on these months.

The main estimand is accounts unseen in the fitted model's March training and May validation cohorts; remove accounts seen in the original June evaluation and in any earlier extension month as well, so repeated test accounts do not cross periods. Use metadata-valid earlier records in supplied March/May/June and newly evaluated July files for this exclusion, including records excluded for body quality. A stricter, separately reported subset requires no observed selected-community activity in any parseable earlier RMHD raw CSV record from 2019 onward. This changes the selection population and does not identify genuinely new Reddit users. Report metadata parsing coverage and missing/unsupported archive files.

Do not remove March training data based on future text or later affiliation. Apply the five-word-shingle Jaccard >=0.8 audit against March/May inputs and report a frozen-model sensitivity after excluding affected later-period authors. August counts/coverage are descriptive; if either class has fewer than 100 accounts, report descriptive metrics and flag imprecision without pooling months or changing exclusions to increase a score.

## Frozen models and reporting

Use the already retained linguistic LR, TF-IDF LR and nonlinear 13-feature histogram-boosting model. Keep May-selected thresholds fixed: mentalhealth LR 0.44, TF-IDF 0.42, boosting 0.45; depression LR 0.37, TF-IDF 0.37, boosting 0.42. Also retain the original 0.5 operating point. All weights, vocabulary, scaling and parameters remain unchanged.

Primary extension contrast is TF-IDF minus linguistic LR macro-F1, with 1,000 paired stratified-account bootstrap replicates, seed 42, separately for each comparison/month. Report all metrics, prevalences and confusion matrices; include paired contrasts against boosting and fixed-model uncertainty. Report stricter historical-absence subsets as sensitivity analyses. Intervals condition on models and class counts; dependence between comparisons prevents pooling them as independent replications. The extension cannot establish clinical anxiety, diagnosis specificity, onset, causal context effects or population screening.

## Training resampling, if run

Predefine 20 stratified March-account bootstrap resamples (seeds 100–119), retaining each selected primary LR C and model recipe. Refit scaling/vocabulary and LR on the resampled training accounts, choose operating points on the original May validation accounts using the existing grid, and report the complete distribution of June macro-F1 and lexical-minus-linguistic differences. Do not select the best seed, replace the primary models or describe this distribution as a population confidence interval. This probes training-sample sensitivity, not uncertainty in psychiatric labels or source collection. Keep it separate from the new-period evaluation, which uses the original frozen models.

## Release

Save raw records, source identities, fitted artifacts and individual probabilities outside Git with restrictive permissions. Public artifacts contain only aggregate counts/results, source/code hashes, protocols and figures. The human authors' institutional determination remains separate from software execution and source-license verification.
