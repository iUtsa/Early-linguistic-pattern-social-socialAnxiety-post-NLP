# Early Detection of Anxiety from Social Media Posts (Preparing for Q1 journal submission by Dec)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-preprint-red.svg)](YOUR_PAPER_LINK)
[![Dataset](https://img.shields.io/badge/dataset-Reddit-orange.svg)](https://www.reddit.com/r/Anxiety/)

> **An interpretable NLP system for detecting anxiety patterns in social media text with 89.34% F1 score**

**Arnab Das Utsa** | Computer Science, Stockton University  
📧 utsaa@go.stockton.edu | 🌐 [https://iutsa.vercel.app/contact.html](WEBSITE)

---

## TL;DR

This project detects anxiety disorders from Reddit posts using **13 interpretable linguistic features**. Unlike black-box deep learning models, our system:

- ✅ **89.34% F1 score** with honest author-disjoint evaluation
- ✅ **Keyword-independent** detection validated three ways
- ✅ **Clinically validated** on DAIC-WOZ interview dataset
- ✅ **Early detection** from just 3 posts (88.44% F1)
- ✅ **Interpretable** features grounded in clinical psychology

**Key Finding**: Self-focused attention (first-person pronoun usage) is the **dominant** anxiety marker, with a coefficient **1.6× larger** than any other feature—validating 50+ years of clinical research.

---

## 📊 Quick Results

<p align="center">
  <img src="figures/main_results_summary.png" alt="Main Results" width="800"/>
</p>

<!-- 
TO ADD THIS FIGURE:
1. Create a figure showing:
   - F1 Score: 89.34%
   - Accuracy: 89.98%
   - Precision: 93.02%
   - Recall: 85.94%
   - ROC-AUC: 0.95
Use a clean bar chart or card layout
Save as: figures/main_results_summary.png
-->

| Metric | Score | 95% CI |
|--------|-------|--------|
| **F1 Score** | **89.34%** | [89.03, 89.66] |
| **Accuracy** | 89.98% | [89.69, 90.27] |
| **Precision** | 93.02% | [92.66, 93.37] |
| **Recall** | 85.94% | [85.48, 86.40] |
| **ROC-AUC** | 0.9500 | [0.9476, 0.9524] |

---

## Why This Matters

### The Problem

- **284 million** people globally have anxiety disorders
- **50-70%** remain undiagnosed due to cost, stigma, and access barriers
- Millions seek support in online communities like r/Anxiety

### The Challenge

Existing NLP approaches have critical flaws:

1. ❌ **Data leakage**: Same users in train & test sets
2. ❌ **Keyword dependence**: Just memorizing "anxiety", "panic"
3. ❌ **Black-box models**: No interpretability for clinicians
4. ❌ **Domain-specific**: Don't generalize to clinical settings

### Our Solution

✅ **Author-disjoint** evaluation (zero user overlap)  
✅ **Three-way validation** proving keyword independence  
✅ **Interpretable features** grounded in psychology  
✅ **Cross-domain validation** on clinical interviews  

---

## System Architecture

<p align="center">
  <img src="figures/system_architecture.png" alt="System Architecture" width="900"/>
</p>

<!--
TO ADD THIS FIGURE:
Create a flowchart showing:
1. Reddit Posts → 2. Feature Extraction (13 features) → 3. Logistic Regression → 4. Anxiety Detection
Include icons for each step
Save as: figures/system_architecture.png
-->

### Pipeline Overview
```
Raw Reddit Post
      ↓
Feature Extraction (13 features)
  ├─ Sentiment (6): VADER + TextBlob
  ├─ Self-Reference (2): Pronoun usage
  └─ Text Structure (5): Length, punctuation, emojis
      ↓
User-Level Aggregation (mean across posts)
      ↓
Logistic Regression (L2 regularization)
      ↓
Prediction: Anxiety (1) or Control (0)
```

---

## 📈 Feature Importance

<p align="center">
  <img src="figures/feature_importance.png" alt="Feature Importance" width="800"/>
</p>

<!--
TO ADD THIS FIGURE:
This should already exist from your training!
If you ran: python src/run.py --mode full --model lr
It's saved at: results/reports/feature_importance.png
Copy it to: figures/feature_importance.png
-->

### Top 10 Features

| Rank | Feature | Coefficient | Direction | Meaning |
|------|---------|-------------|-----------|---------|
| 🥇 1 | **First-person pronoun rate** | **+4.11** | ↑ Anxiety | "I", "me", "my" usage |
| 🥈 2 | VADER neutral | -2.65 | ↓ Anxiety | Emotional vs neutral tone |
| 🥉 3 | VADER positive | -1.61 | ↓ Anxiety | Positive emotion level |
| 4 | Punctuation density | -1.40 | ↓ Anxiety | Structured writing |
| 5 | Avg word length | +0.86 | ↑ Anxiety | Vocabulary complexity |
| 6 | VADER negative | +0.63 | ↑ Anxiety | Negative emotion |
| 7 | TextBlob subjectivity | +0.60 | ↑ Anxiety | Opinion vs fact |
| 8 | Pronoun count | +0.53 | ↑ Anxiety | Raw self-reference |
| 9 | TextBlob polarity | -0.40 | ↓ Anxiety | Sentiment valence |
| 10 | Emoji count | -0.22 | ↓ Anxiety | Emoji usage |

**Key Insight**: First-person pronoun rate is **1.6× stronger** than any other feature! This validates decades of clinical research on self-focused attention in anxiety. 🎯

---

## Three-Way Keyword Robustness Validation

We prove the model learns **genuine psychological patterns**, not just keywords:

<p align="center">
  <img src="figures/keyword_robustness.png" alt="Keyword Robustness" width="800"/>
</p>

<!--
TO ADD THIS FIGURE:
Create a bar chart showing:
- Full Model: 89.34% F1
- Without Sentiment Features: 88.69% F1 (only -0.65pp drop)
- Keyword Masking: 88.70% F1 (only -0.64pp drop)
- Pronoun-Only: 87.01% F1 (still excellent!)
Add annotations showing small drops prove keyword independence
Save as: figures/keyword_robustness.png
-->

### The Tests

| Test | Method | F1 Score | Δ F1 | Conclusion |
|------|--------|----------|------|------------|
| **Full Model** | All 13 features | 89.34% | baseline | - |
| **Test 1: Ablation** | Remove sentiment features | 88.69% | -0.65pp | ✅ Minimal drop |
| **Test 2: Masking** | Replace keywords with [MASK] | 88.70% | -0.64pp | ✅ Minimal drop |
| **Test 3: Pronoun-Only** | Use only self-reference | 87.01% | -2.33pp | ✅ Still strong! |

**Result**: All three independent tests converge—the model relies on **self-focused attention** (pronouns), not disorder vocabulary! 🎉

---

## 🏥 Cross-Domain Validation

Does Reddit learning generalize to clinical settings?

<p align="center">
  <img src="figures/cross_domain_validation.png" alt="Cross-Domain Validation" width="800"/>
</p>

<!--
TO ADD THIS FIGURE:
Create a comparison showing:
Left side: Reddit (r/Anxiety) - informal social media
Right side: DAIC-WOZ - clinical interviews
Middle: Show 75% feature consistency with arrows
Bottom: Show 3 features with large effects in both domains:
  - VADER negative (↑ anxiety in both)
  - VADER positive (↓ anxiety in both)
  - TextBlob polarity (↓ anxiety in both)
Save as: figures/cross_domain_validation.png
-->

### Validation on DAIC-WOZ Clinical Interviews

We tested our features on professional clinical interviews (189 participants):

| Feature | Reddit (Cohen's d) | Clinical (Hedges' g) | Consistent? |
|---------|-------------------|---------------------|-------------|
| **VADER negative** | +0.42 | +0.85 (large) | ✅ Both ↑ |
| **VADER positive** | -0.38 | -0.92 (large) | ✅ Both ↓ |
| **TextBlob polarity** | -0.45 | -0.78 (large) | ✅ Both ↓ |

**Result**: 75% of features show consistent patterns across domains! Patterns learned from Reddit **generalize to clinical interviews**.

---

## ⚡ Early Detection

Can we detect anxiety from just the first few posts?

<p align="center">
  <img src="figures/early_detection.png" alt="Early Detection" width="800"/>
</p>

<!--
TO ADD THIS FIGURE:
Create a line graph showing:
X-axis: Number of posts (3, 5, 10, All)
Y-axis: F1 Score (85-90%)
Points: 88.44%, 88.14%, 87.75%, 89.34%
Add annotations:
  - "3 posts ≈ First week"
  - "Only -0.90pp drop with 3 posts!"
Save as: figures/early_detection.png
-->

### Results

| Posts Used | F1 Score | Δ from Full | Timeline |
|------------|----------|-------------|----------|
| **3 posts** | **88.44%** | **-0.90pp** | ≈ 1 week |
| 5 posts | 88.14% | -1.20pp | ≈ 2 weeks |
| 10 posts | 87.75% | -1.59pp | ≈ 1 month |
| All posts | 89.34% | baseline | - |

**Conclusion**: Just **3 posts** achieves 88.44% F1! Early detection is feasible within **days** of someone seeking online support. ⚡

---

## 📁 Project Structure
```
Early-linguistic-pattern-social-socialAnxiety-post-NLP/
├── README.md                          # You are here!
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
│
├── src/                               # Core source code
│   ├── run.py                         # Main pipeline orchestrator
│   ├── train.py                       # Model training
│   ├── evaluate.py                    # Model evaluation
│   ├── features.py                    # Feature extraction (13 features)
│   ├── data_loader.py                 # Data loading & validation
│   ├── dataset_prep.py                # Dataset preparation
│   ├── plots.py                       # Visualization utilities
│   ├── stats.py                       # Statistical tests
│   ├── explain.py                     # SHAP explanations
│   └── utils.py                       # Helper functions
│
├── scripts/                           # Utility scripts
│   ├── build_rmhd.py                  # Build Reddit dataset
│   ├── test_external.py               # Cross-domain validation
│   ├── keyword_lists.py               # Keyword definitions
│   └── annotate_pseudo_onset.py       # Temporal annotation
│
├── live_demo.py                       # 🚀 Interactive prediction demo
├── config.yaml                        # Configuration file
│
├── figures/                           # 📊 Result visualizations
│   ├── main_results_summary.png
│   ├── system_architecture.png
│   ├── feature_importance.png
│   ├── keyword_robustness.png
│   ├── cross_domain_validation.png
│   ├── early_detection.png
│   └── roc_curve.png
│
├── models/                            # 🤖 Trained models (not in repo)
│   └── logistic_regression_fixed.pkl  # Main model (150KB)
│
├── data/                              # 📁 Datasets (not in repo - too large)
│   ├── raw/                           # Raw Reddit posts
│   ├── processed/                     # Processed features
│   └── external/                      # DAIC-WOZ validation data
│
└── results/                           # 📈 Output files (generated)
    ├── val_metrics.csv                # Validation metrics
    ├── early_slice.csv                # Early detection results
    └── reports/                       # Detailed reports
```

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/iUtsa/Early-linguistic-pattern-social-socialAnxiety-post-NLP.git
cd Early-linguistic-pattern-social-socialAnxiety-post-NLP

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Interactive Demo

Try the live prediction system:
```bash
python3 live_demo.py
```

**Demo Options:**
```
1. Batch predict from data/demo/ (automatic)
2. Interactive mode (type your own text)
3. Demo with built-in examples
4. Exit
```

**Example Session:**
```
Enter text (press Enter twice when done):
I feel so anxious all the time. I can't stop worrying about everything.
I keep thinking about what people think of me.

================================================================================
PREDICTION RESULT
================================================================================

Input Text:
  "I feel so anxious all the time. I can't stop worrying..."

🎯 Prediction: ANXIETY
   Confidence: 92.4%

📊 Probability Breakdown:
   Control: 7.6%
   Anxiety: 92.4%

🔑 KEY FEATURES DETECTED:
  ⚠️  High self-reference: 3 first-person pronouns (16.7% of words)
  ⚠️  Elevated negative sentiment: 0.34
  ⚠️  Low positive sentiment: 0.00

  → Combined pattern: High self-focus + negative emotion
     This is a core marker of anxiety expression
================================================================================
```

---

## 🔬 Training Your Own Model

### Full Training Pipeline
```bash
cd src

# Train on full dataset
python3 run.py --mode full --model lr --config ../config.yaml
```

**Output:**
- Trained model: `models/checkpoints/lr_full.pkl`
- Metrics: `results/val_metrics.csv`
- Feature importance: `results/reports/feature_importance.png`
- SHAP analysis: `results/reports/shap_summary.png`

### Early Detection Experiments
```bash
# Test with k={3,5,10} posts per user
python3 run.py --mode early_slice --model lr
```

**Output:**
- Results: `results/early_slice.csv`
- Plot: `results/reports/early_slice_f1.png`

### Cross-Domain Validation
```bash
# Validate on DAIC-WOZ clinical interviews
python3 run.py --mode stage_compare --stage_csv data/staged_posts.csv
```

---

## 📊 Reproducing Paper Results

### Main Results (Table 4)
```bash
cd src
python3 run.py --mode full --model lr
```

Check: `results/val_metrics.csv`

### Feature Importance (Table 5, Figure 6)
```bash
python3 train.py  # Displays coefficients
```

Check: `results/reports/feature_importance.png`

### Keyword Robustness (Table 6-8)
```bash
python3 keyword_masking_ablation_fixed.py
```

Check: `results/keyword_ablation_results.csv`

### Cross-Domain Validation (Section 6)
```bash
cd scripts
python3 test_external.py --dataset daic_woz
```

Check: `results/cross_domain_validation.txt`

### Early Detection (Table 9)
```bash
cd src
python3 run.py --mode early_slice
```

Check: `results/early_slice.csv`

---

## 📚 Dataset

### Reddit Mental Health Dataset (RMHD)

- **Source**: Reddit r/Anxiety + control subreddits
- **Period**: January 2018 - December 2019
- **Posts**: 286,994 (50% anxiety, 50% control)
- **Users**: 155,599 (author-disjoint splits)

**⚠️ Dataset not included** in this repository due to size (>2GB) and privacy considerations.

### Data Splits

| Split | Posts | Users | Anxiety | Control |
|-------|-------|-------|---------|---------|
| Train | 201,646 | 128,140 | 100,823 | 100,823 |
| Val | 42,703 | 22,459 | 21,352 | 21,351 |
| Test | 42,645 | 27,459 | 21,322 | 21,323 |

**Key**: Zero user overlap between splits (author-disjoint)

### Getting the Data

Due to Reddit's Terms of Service and privacy considerations, we cannot directly distribute the dataset. However, you can:

1. **Use your own Reddit data**: Collect from r/Anxiety using Reddit API
2. **Request access**: Email arnab.utsa@stockton.edu for research purposes
3. **Use public datasets**: DAIC-WOZ, CLPsych shared tasks

---

## 🔧 Configuration

Edit `config.yaml` to customize:
```yaml
dataset:
  posts_csv: "data/raw/final_dataset_fixed.csv"
  users_csv: null

training:
  seed: 42
  results_dir: "results"
  reports_dir: "results/reports"
  early_slice_ks: [3, 5, 10]
  
features:
  use_sentiment: true
  use_pronouns: true
  use_structure: true
```

---

## 📦 Requirements

**Python Version**: 3.8+

**Core Dependencies**:
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
vaderSentiment>=3.3.2
textblob>=0.17.1
emoji>=2.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
lightgbm>=4.0.0
pyyaml>=6.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎓 Citation

If you use this work in your research, please cite:
```bibtex
@article{utsa2024anxiety,
  title={Early Detection of Anxiety Disorders from Social Media Text Using Interpretable Linguistic Features},
  author={Utsa, Arnab Das},
  journal={arXiv preprint},
  year={2024},
  url={https://github.com/iUtsa/Early-linguistic-pattern-social-socialAnxiety-post-NLP}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2024 Arnab Das Utsa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for contribution:**
- Multilingual support
- Additional feature engineering
- Deep learning baselines
- Visualization improvements
- Documentation enhancements

---

## ⚠️ Ethical Considerations

### This is NOT a diagnostic tool

- ✅ **Use**: Pre-screening to identify individuals who may benefit from professional evaluation
- ✅ **Use**: Research to understand linguistic markers of anxiety
- ❌ **Don't**: Replace clinical diagnosis or professional treatment
- ❌ **Don't**: Make decisions about individuals without human oversight

### Privacy & Consent

- All data from publicly available Reddit posts
- Usernames anonymized
- No personally identifiable information collected
- Used under Reddit's Terms of Service for research

### Responsible AI Principles

1. **Interpretability**: 13 explainable features grounded in psychology
2. **Validation**: Three-way robustness testing + cross-domain validation
3. **Transparency**: Open methodology and code
4. **Human-in-loop**: Designed to assist clinicians, not replace them
5. **Continuous monitoring**: Regular audits for fairness and accuracy

### Limitations

- Trained on Reddit data (may not generalize to other platforms)
- English only (no multilingual validation)
- Binary classification (doesn't capture severity or subtypes)
- Self-selection bias (users in r/Anxiety may differ from general anxiety population)

---

## 📞 Contact

**Arnab Das Utsa**  
Computer Science Department, Stockton University

- 📧 Email: utsaa@go.stockton.edu
- 🌐 Website: [https://iutsa.vercel.app/index.html](WEBSITE)
- 💼 LinkedIn: [https://www.linkedin.com/in/iutsa24/](LINKEDIN)
- 🐦 Twitter: [@iADUtsa](TWITTER)

**Questions?** Open an issue or email me!

---

## 🌟 Acknowledgments

- **Dataset**: Reddit Mental Health Dataset (RMHD)
- **Validation**: DAIC-WOZ corpus (USC)
- **Tools**: scikit-learn, VADER, TextBlob
- **Advisors**: Dr. Sujoy Charkaborty
- **Institution**: Stockton University Computer Science Department

---

## 📈 Project Status

- ✅ Core system complete (89.34% F1)
- ✅ Three-way keyword validation
- ✅ Cross-domain validation on DAIC-WOZ
- ✅ Early detection experiments
- 🚧 Multilingual extension (in progress)
- 🚧 Temporal modeling (planned)
- 🚧 Real-time deployment (planned)

---

## 🔗 Related Projects

- [CLPsych Shared Tasks](http://clpsych.org/)
- [Mental Health NLP Resources](https://github.com/psychiatric-nlp)
- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/)

---

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/iUtsa/Early-linguistic-pattern-social-socialAnxiety-post-NLP?style=social)
![GitHub forks](https://img.shields.io/github/forks/iUtsa/Early-linguistic-pattern-social-socialAnxiety-post-NLP?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/iUtsa/Early-linguistic-pattern-social-socialAnxiety-post-NLP?style=social)

---

<p align="center">
  <b>If you find this useful, please star the repo! ⭐</b>
</p>

<p align="center">
  First author:  Arnab Das Utsa
</p>

<p align="center">
  <sub>© 2025 Arnab Das Utsa. All rights reserved.</sub>
</p>