# Data card: RMHD version 1 used in the revised study

Verified 7 October 2026. The revised outcome is **exclusive observed community affiliation**, not psychiatric diagnosis, clinical anxiety severity or onset.

## Identified upstream release

- Dataset: [Reddit Mental Health Dataset (RMHD)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset), Kaggle dataset ID 3687885, version **1**, modified 1 September 2023.
- Source paper: [Rani, Ahmed and Subramani (2024)](https://doi.org/10.3390/app14041547).
- The source paper's short link resolves to the dataset above. Its dataset landing-page JSON-LD and Kaggle API both list **CC0: Public Domain**. This is the dataset publisher's stated license, rather than an inference from the article's license.
- The complete version-1 file listing contains **225 unique file paths**, with total listed file sizes 1,681,516,066 bytes. Pagination was exhausted. Raw Part A spans January 2019–August 2022; annotated Part B contains four root-cause categories, not diagnoses.
- All **15 supplied authored files** are now verified byte-for-byte against extracted version-1 source files using SHA-256. These comprise March, May and June 2022 exports. The upstream release also includes both identical-looking June mentalhealth filenames and no June SuicideWatch file in that folder. The duplicate in the uploads therefore has an upstream counterpart.
- The complete archive was downloaded on 7 October 2026: 647,220,389 compressed bytes, SHA-256 `6078fe83304c266ca976973f8b1e553dc5d818c59810508151f9e6bc615bf9e4`. All 225 extracted files agree with the upstream listing and have content hashes. This resolves the earlier blocked archive access and upload-identity limitation.
- The archive contains 220 raw paths: 219 schema-compatible CSVs and one Apple Numbers file (`depmay21.numbers`). CSVs contain 1,851,580 rows before removing two byte-identical file duplicates, and 1,834,871 afterward. These are raw export rows, not verified unique original posts. Metadata-valid rows used for history total 1,813,026. These counts use this release and policy and should not be substituted for the source article’s reported 1,494,019 posts.
- The two duplicate pairs are December 2019 depression/SuicideWatch filenames and June 2022 mentalhealth filenames. Actual observed community fields define labels; filenames do not. The Numbers file is explicitly outside CSV history coverage. No claim of complete observed lifetime activity is made.

See [source metadata and file correspondence](rmhd_source_manifest.json). Publisher responses and the source PDF were retained privately/under ignored literature paths; hashes and reading scope are recorded in [source_manifest.json](source_manifest.json).

## Included observations and selection

Two fixed contrasts use Anxiety versus mentalhealth and Anxiety versus depression. March trains; May tunes; June supplies the original evaluation. Frozen July/August models additionally evaluate accounts absent from supplied earlier selected-community records and earlier extension months, with a separate stricter historical-absence sensitivity over the parseable full release. All timestamps use numeric UTC fields. Original post IDs are absent. Authors are observed Reddit accounts, not verified independent people. Mixed selected-community affiliation within a month is excluded; exclusivity does not mean absence of comorbidity or activity elsewhere.

Bodies must contain at least ten cleaned words. Titles and bodies are combined. Metadata/placeholder checks, byte-identical-file removal and exact record/text deduplication precede analysis. Author and exact-text overlap across model splits is zero. The additional near-duplicate audit uses an explicitly limited verbatim-reuse definition. Full cohort flow and class counts are in [uploaded_pilot_results.json](uploaded_pilot_results.json).

General-interest files without author identifiers and the clinical PHQ-8 files are not part of these community-classification experiments. No fabricated control identities or depression-to-anxiety relabelling enter the new outcomes.

## Use, ethics and release

The study author supplied the files and requested analysis. The now-identified source states CC0; the current project must still document its applicable institutional determination and responsible research use. The author reports that a determination exists; its reviewing body, decision, reference, date and scope await documentation. The source paper's Victoria University HREC HRE23-005 approval belongs to that source study and is not assigned to this analysis. No current-project approval number or exemption is invented.

Public research artifacts comprise code, frozen protocols, aggregate counts/results, input/code hashes and aggregate figures. Raw texts, usernames, clinical participant-level records, fitted vocabularies and individual predictions remain private. A future researcher can obtain the listed upstream release and apply the documented pipeline subject to their own applicable requirements. The present release does not provide a new downloadable clinical dataset.

## Scope limits

These are English-oriented lexicon features, without a verified per-post language audit. Bots beyond the explicit placeholder policy and multiple accounts remain possible. Five selected 2022 months share one collection process; full-release records provide historical exclusions, not independent-source validation. The two contrasts share some positive accounts. Neither clinical discrimination, diagnosis-specific linguistic mechanisms, prospective onset nor population screening utility is established.
