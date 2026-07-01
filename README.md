# Engineering College Review Analysis — Data Science Project

A comprehensive data science project that collects, cleans, integrates, and analyzes **17,000+ student reviews** of **150 Indian engineering colleges** from multiple platforms. The project applies **sentiment analysis**, **topic extraction**, **machine learning models** (regression & classification), and **advanced visualizations** to uncover actionable insights about student perception of higher education institutions.

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Data Sources](#-data-sources)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Data Collection](#-data-collection)
- [Data Preparation & Feature Engineering](#-data-preparation--feature-engineering)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Machine Learning Models](#-machine-learning-models)
- [Visualizations](#-visualizations)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Assumptions](#-assumptions)

---

## Problem Statement

Student reviews across platforms such as **CollegeDunia**, **Careers360**, and **Kaggle** contain valuable insights about institutional quality and student satisfaction. However, these reviews are:

- **Fragmented** across multiple sources
- **Inconsistent** in format (different column schemas, rating scales, date formats)
- A mix of **structured ratings** and **unstructured text**

This project integrates multi-source review data and analyzes student perception of engineering colleges using both numerical ratings and textual sentiment.

---

## Objectives

1. Merge and standardize review datasets from multiple sources.
2. Clean and preprocess textual and numerical data.
3. Analyze rating distributions across colleges and platforms.
4. Perform sentiment analysis on review text.
5. Evaluate the impact of key topics (placements, faculty, infrastructure, etc.) on ratings.
6. Build ML models to **predict ratings** (regression) and **classify rating categories** (classification).
7. Visualize institutional perception using advanced EDA techniques.

---

## Data Sources

| Source | Reviews | Collection Method |
|---|---|---|
| **CollegeDunia** | ~1,500 | Scroll-based web scraping (Selenium) |
| **Careers360** | ~11,368 | Paginated web scraping via iterative requests |
| **Kaggle** | ~4,307 | Downloaded & merged 2021–2023 datasets |
| **Combined (after cleaning)** | **17,131** | — |

150 colleges were selected and assigned unique IDs for cross-source merging.

---

## Project Structure

```
Data Science Project/
│
├── README.md                                  # This file
│
├── Review_1.ipynb                             # Data collection, cleaning, EDA & feature engineering
├── Review_2.ipynb                             # ML models (regression + classification) & visualizations
│
├── Data Collection/                           # Raw scraping scripts & intermediate data
│   ├── webScrap.py                            # Example scraping script (BeautifulSoup)
│   ├── Careers360/                            # Careers360 scraper, raw data, cleaning notebook
│   │   ├── WebScrap.py
│   │   ├── Cleaning.ipynb
│   │   ├── careers360_raw_reviews.csv
│   │   └── careers360_final_dataset.csv
│   ├── CollegeDunia/                          # CollegeDunia scraper, raw data, mapping files
│   │   ├── scripts.ipynb
│   │   ├── collegedunia_reviews_scraped.csv
│   │   └── map_final_collegedunia.csv
│   └── Kaggle/                                # Kaggle dataset scripts, merged & cleaned files
│       ├── Scripts.ipynb
│       ├── collegereview2021.csv
│       ├── collegereview2022.csv
│       ├── collegereview2023.csv
│       └── kaggle_reviews_cleaned.csv
│
├── collegedunia.csv                           # Cleaned per-source datasets
├── careers360.csv
├── kaggle.csv
├── final_college_reviews_dataset.csv          # Final merged & feature-engineered dataset (17,131 rows × 18 cols)
│
├── review2_best_regression_model.joblib       # Saved best regression model (RandomForest)
├── review2_best_classification_model.joblib   # Saved best classification model (RandomForest)
├── review2_regression_metrics.csv             # Regression metrics comparison
├── review2_classification_metrics.csv         # Classification metrics comparison
└── review2_best_model_predictions.csv         # Predictions from the best model
```

---

## Pipeline Overview

```
  ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
  │ Web Scraping │───▶│  Data Cleaning │────▶│   Merging &   │
  │ (3 sources)  │     │  & Formatting  │     │ Standardizing │
  └──────────────┘     └────────────────┘     └───────┬───────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │   Feature     │
                                              │  Engineering  │
                                              └───────┬───────┘
                                                      │
                              ┌───────────────────────┼────────────────────────┐
                              ▼                       ▼                        ▼
                      ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
                      │     EDA      │        │  ML Models   │        │Visualization │
                      │  (Review 1)  │        │  (Review 2)  │        │  (Review 2)  │
                      └──────────────┘        └──────────────┘        └──────────────┘
```

---

## Data Collection

Three data sources were scraped/downloaded and pre-processed in `Data Collection/`:

- **CollegeDunia**: Reviews scraped using Selenium (scroll-based loading). Ratings were embedded in strings (e.g., `"4.3Helpful"`) and required regex extraction.
- **Careers360**: Reviews scraped via paginated HTTP requests. Dates formatted as `"Posted on 19 Dec '25 by ..."` required custom parsing.
- **Kaggle**: Three annual CSV files (2021–2023) were downloaded and merged. Ratings were on a 0–5 scale.

College IDs and mapping files were used to unify institution identities across sources.

---

## Data Preparation & Feature Engineering

**Implemented in [Review_1.ipynb](Review_1.ipynb):**

### Data Cleaning
- Fixed BOM encoding issues in CollegeDunia columns
- Extracted numeric ratings from strings like `"4.8Helpful"` using regex
- Standardized date formats across all three sources
- Unified rating column names and scales (all normalized to 0–5)
- Removed rows with missing `review_text` or `rating`
- Filtered out rating outliers (`< 0` or `> 5`)
- Lowercased and whitespace-normalized all review text

### Feature Engineering (18 features total)

| Feature | Description |
|---|---|
| `review_length` | Word count of the review text |
| `rating_category` | Categorical bucket: **Low** (≤2), **Medium** (2–4), **High** (>4) |
| `year` | Extracted from the review date |
| `sentiment_score` | VADER compound score (−1 to +1) |
| `sentiment_label` | Positive / Neutral / Negative (thresholded at ±0.05) |
| `placement` | Boolean — review mentions placement-related keywords |
| `faculty` | Boolean — review mentions faculty-related keywords |
| `infrastructure` | Boolean — review mentions infrastructure keywords |
| `hostel` | Boolean — review mentions hostel/accommodation keywords |
| `campus` | Boolean — review mentions campus/environment keywords |
| `weight` | Weighted average rating per college for variance-aware scoring |

### Sentiment Analysis (VADER)
- Used **VADER** (Valence Aware Dictionary for Sentiment Reasoning), a rule-based sentiment tool designed for social text.
- The compound score captures overall emotional tone:
  - `≥ 0.05` → **Positive**
  - `≤ −0.05` → **Negative**
  - Otherwise → **Neutral**
- Distribution: **97.2%** Positive | **2.3%** Negative | **0.5%** Neutral

---

## Exploratory Data Analysis (EDA)

**Conducted in [Review_1.ipynb](Review_1.ipynb) (Level 1 EDA):**

Visualizations include:
- **Rating distribution** histogram
- **Rating distribution across platforms** (violin plot) — Careers360 shows strongest positive bias; CollegeDunia and Kaggle show wider spread
- **Rating vs. Sentiment density plot** — confirms alignment between numeric ratings and emotional tone
- **Sentiment distribution** by source
- **Reviews per year** timeline
- **Rating category distribution** (High: 10,500 | Medium: 6,568 | Low: 63)
- **Topic mention frequency** analysis (placements, faculty, infrastructure, hostel, campus)
- **Weighted average rating** per college

---

## Machine Learning Models

**Implemented in [Review_2.ipynb](Review_2.ipynb):**

### Preprocessing Pipeline
- **Text features**: TF-IDF vectorization (max 5,000 features, 1–2 ngrams, min_df=3, English stopwords)
- **Categorical features**: SimpleImputer + OneHotEncoder
- **Numeric features**: SimpleImputer (median) + StandardScaler
- Train/Test split: **80/20** (random_state=42)

### Part A — Regression (Predict rating on a 0–5 scale)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **RandomForest** | **0.383** | **0.495** | **0.384** |
| Ridge | 0.407 | 0.515 | 0.335 |
| GradientBoosting | 0.427 | 0.523 | 0.314 |
| Baseline (Dummy) | 0.531 | 0.631 | ~0.000 |

**Best model: RandomForest Regressor** (250 estimators)
- Within ±0.25 stars: **41.64%**
- Within ±0.50 stars: **71.87%**
- Within ±1.00 stars: **96.35%**

### Part B — Classification (Predict rating category: Low / Medium / High)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **RandomForest** | **0.811** | **0.807** |
| LogisticRegression | 0.778 | 0.774 |
| Baseline (Dummy) | 0.614 | 0.467 |

**Best model: RandomForest Classifier** (250 estimators)

All best models are saved as `.joblib` files for reuse.

---

## Visualizations

**Implemented in [Review_2.ipynb](Review_2.ipynb) using Matplotlib, Seaborn, and Plotly:**

### Static Visualizations (Matplotlib / Seaborn)
- Regression model comparison bar chart (MAE, RMSE, R²)
- Residual distribution plot for best regression model
- Confusion matrix heatmap for best classification model

### Interactive Visualizations (Plotly)
- Average rating by platform (color-coded by review count)
- Rating distribution by platform (box plot)
- Rating distribution by sentiment label (box plot)
- Sentiment score distribution (histogram)
- Rating vs. sentiment score (scatter plot with regression line)
- Reviews per year (bar chart)
- Topic mention frequency (bar chart)
- Top 20 colleges by weighted average rating (horizontal bar chart)

---

## Key Results

| Insight | Detail |
|---|---|
| **Dataset Size** | 17,131 reviews across 150 colleges from 3 sources |
| **Sentiment Skew** | 97.2% positive reviews — strong positive bias in student feedback |
| **Best Regression Model** | RandomForest — MAE of 0.38, predicts within ±0.5 stars 72% of the time |
| **Best Classification Model** | RandomForest — 81.1% accuracy, 0.807 weighted F1 |
| **Platform Bias** | Careers360 has the highest positive rating concentration; Kaggle shows more critical reviews |
| **Top Topics** | Placements and campus life are the most frequently mentioned aspects |
| **Average Review Length** | ~163 words |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.13 |
| **Data Processing** | Pandas, NumPy |
| **Web Scraping** | Requests, BeautifulSoup, Selenium |
| **NLP / Sentiment** | VADER (vaderSentiment) |
| **Machine Learning** | scikit-learn (Ridge, RandomForest, GradientBoosting, LogisticRegression, DummyRegressor) |
| **Feature Engineering** | TF-IDF (TfidfVectorizer), OneHotEncoder, StandardScaler |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Model Persistence** | Joblib |
| **Environment** | Jupyter Notebook |

---

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn vaderSentiment joblib requests beautifulsoup4
```

### Running the Notebooks

1. **Review 1 — Data Collection, Cleaning & EDA:**
   ```
   jupyter notebook Review_1.ipynb
   ```
   This notebook loads the three source CSVs (`collegedunia.csv`, `careers360.csv`, `kaggle.csv`), cleans and merges them, performs feature engineering, runs EDA, and exports the final dataset (`final_college_reviews_dataset.csv`).

2. **Review 2 — Modeling & Visualization:**
   ```
   jupyter notebook Review_2.ipynb
   ```
   This notebook loads `final_college_reviews_dataset.csv`, trains regression and classification models, evaluates performance, generates visualizations, and saves the best models.

---

## Assumptions

1. Student reviews represent genuine experiences.
2. Normalized ratings across platforms are comparable.
3. VADER sentiment analysis accurately captures emotional tone of review text.
4. Topic mentions (placement, faculty, etc.) reflect important aspects influencing ratings.

---

> **Note:** This project was developed as part of an academic Data Science course and is structured around two review milestones — data preparation/EDA (Review 1) and modeling/visualization (Review 2).
