# Data card: RMHD examples used in the revised study

Verified 7 October 2026. The revised outcome is **exclusive observed community affiliation**, not psychiatric diagnosis, clinical anxiety severity or onset.

## Identified upstream release

- Dataset: [Reddit Mental Health Dataset (RMHD)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset), Kaggle dataset ID 3687885, version **1**, modified 1 September 2023.
- Source paper: [Rani, Ahmed and Subramani (2024)](https://doi.org/10.3390/app14041547).
- The source paper's short link resolves to the dataset above. Its dataset landing-page JSON-LD and Kaggle API both list **CC0: Public Domain**. This is the dataset publisher's stated license, rather than an inference from the article's license.
- The complete version-1 file listing contains **225 unique file paths**, with total listed file sizes 1,681,516,066 bytes. Pagination was exhausted. Raw Part A spans January 2019–August 2022; annotated Part B contains four root-cause categories, not diagnoses.
- All **15 supplied authored filenames** have corresponding version-1 paths with identical listed byte lengths. These comprise March, May and June 2022 exports. The upstream release also includes both identical-looking June mentalhealth filenames and no June SuicideWatch file in that folder. The duplicate in the uploads therefore has an upstream counterpart.
- **Filename/byte-length correspondence is not a checksum identity test.** Full archive retrieval redirects to a cloud-storage host currently blocked by the execution network; upstream file hashes and byte identity have not been independently verified. The uploaded contents themselves have SHA-256 hashes in the aggregate manifest. Their completeness relative to upstream files is corroborated by byte lengths, not proven.

See [source metadata and file correspondence](rmhd_source_manifest.json). Publisher responses and the source PDF were retained privately/under ignored literature paths; hashes and reading scope are recorded in [source_manifest.json](source_manifest.json).

## Included observations and selection

Two fixed contrasts use Anxiety versus mentalhealth and Anxiety versus depression. March trains; May tunes; June evaluates previously unobserved accounts in supplied selected-community records. All timestamps use numeric UTC fields. Original post IDs are absent. Authors are observed Reddit accounts, not verified independent people. Mixed selected-community affiliation within a month is excluded; exclusivity does not mean absence of comorbidity or activity elsewhere.

Bodies must contain at least ten cleaned words. Titles and bodies are combined. Metadata/placeholder checks, byte-identical-file removal and exact record/text deduplication precede analysis. Author and exact-text overlap across model splits is zero. The additional near-duplicate audit uses an explicitly limited verbatim-reuse definition. Full cohort flow and class counts are in [uploaded_pilot_results.json](uploaded_pilot_results.json).

General-interest files without author identifiers and the clinical PHQ-8 files are not part of these community-classification experiments. No fabricated control identities or depression-to-anxiety relabelling enter the new outcomes.

## Use, ethics and release

The study author supplied the files and requested analysis. The now-identified source states CC0; the current project must still document its applicable institutional determination and responsible research use. The source paper's Victoria University HREC HRE23-005 approval belongs to that source study. No approval/exemption for this secondary analysis has been supplied or invented.

Public research artifacts comprise code, frozen protocols, aggregate counts/results, input/code hashes and aggregate figures. Raw texts, usernames, clinical participant-level records, fitted vocabularies and individual predictions remain private. A future researcher can obtain the listed upstream release and apply the documented pipeline subject to their own applicable requirements. The present release does not provide a new downloadable clinical dataset.

## Scope limits

These are English-oriented lexicon features, without a verified per-post language audit. Bots beyond the explicit placeholder policy and multiple accounts remain possible. The time windows are short and share one collection process. The two contrasts share some positive accounts. Neither clinical discrimination, diagnosis-specific linguistic mechanisms, prospective onset nor population screening utility is established.
