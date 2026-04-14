# Airbnb Review Sentiment Modeling (2019 and 2023)

## Notebooks Covered
- `Revs_19.ipynb`
- `Revs_23.ipynb`

## Goal
These two notebooks build a year-specific sentiment analysis and text classification pipeline for Airbnb reviews.
- `Revs_19.ipynb` processes and models 2019 review data.
- `Revs_23.ipynb` processes and models 2023 review data.

Both notebooks follow nearly the same structure and methodology so results can be compared across years.

## High-Level Workflow

### 1. Data Loading and Cleaning (`question_1`)
- Mounts Google Drive and loads monthly review CSV files.
- Concatenates monthly data into one yearly dataframe.
- Cleans `comments` with:
  - lowercasing
  - punctuation removal
  - stopword removal
  - lemmatization
  - alphabetic filtering
- Performs language detection and keeps English reviews.

### 2. Weak Labeling + Transformer Labeling
- Uses positive/negative lexicon files to create initial sentiment-oriented subsets.
- Uses Hugging Face transformer pipeline (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to assign sentiment labels.
- Saves labeled data to yearly CSVs (`sentiment_revs_19.csv` / `sentiment_revs_23.csv`).

### 3. Train/Test Artifact Creation
- Splits data and exports:
  - test sets (`test_revs_19.tsv`, `test_revs_23.tsv`)
  - training sets (`our_train_19.tsv`, `our_train_23.tsv`)

### 4. Feature Engineering (`Question 2`)
- Builds TF-IDF vectors and saves vectorizers (`TF-IDF_VECT_19.pkl`, `TF-IDF_VECT_23.pkl`).
- Trains Word2Vec embeddings and saves models (`MODEL_WORD__2-19.pkl`, `MODEL_WORD__2-23.pkl`).

### 5. Model Training and Evaluation
For each feature type (TF-IDF and Word2Vec), the notebooks train and evaluate:
- SVM
- Random Forest
- KNN

They run:
- train/validation splits (with printed validation accuracy)
- 10-fold stratified cross-validation
- weighted Precision, Recall, F1, and Accuracy

### 6. Prediction Exports
Each model variant writes predicted sentiments to CSV outputs.

### 7. Semantic Neighborhood Analysis (`question_3`)
Using Word2Vec, each notebook computes neighborhood-based similarity measures for user-provided word pairs:
- maximum neighborhood similarity
- correlation of neighborhood similarities
- sum of squared neighborhood similarities

This section is interactive and continues until the user enters `-1`.

## How the Goal Is Achieved
- **Text preprocessing** standardizes noisy review text.
- **Transformer-based sentiment labeling** provides scalable labels.
- **Dual representations** (TF-IDF + embeddings) support both sparse and dense feature spaces.
- **Multiple classifiers** provide model comparison rather than relying on one algorithm.
- **Cross-validation metrics** quantify robustness.
- **Word-level similarity analysis** adds semantic interpretation beyond document classification.

## Inputs and Outputs

### Required Inputs
- Google Drive directory layout for year/month review files.
- `positive-words.txt` and `negative-words.txt` in Drive.

### Common Outputs
- cleaned yearly datasets
- labeled sentiment datasets
- train/test TSV files
- serialized vectorizer and embedding models (`.pkl`)
- per-model prediction CSV files
- sentiment distribution chart (`question_1_updated.png`)

## Dependencies
Main libraries used:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk`, `langdetect`
- `gensim` (Word2Vec)
- `transformers` (Hugging Face pipeline)
- `scikit-learn` (SVM, RF, KNN, metrics, CV)

The notebooks are authored for Colab-style execution (`/content/drive/...` paths and inline `pip install` calls).

## Important Notes and Caveats
- Paths are hard-coded for Colab Drive and should be parameterized for local runs.
- Some steps are computationally expensive (transformer inference over large review sets).
- Lexicon filtering and manual class balancing can influence final label distribution.
- `Revs_23.ipynb` contains a repeated KNN (TF-IDF) block; this is functional but redundant.
- Interactive loops in `question_3` require manual input during execution.

## Recommended Next Improvements
- Move shared logic to reusable Python modules.
- Centralize configuration (paths, sample sizes, model params).
- Add deterministic seeds consistently across all stochastic steps.
- Add notebook-to-script pipeline orchestration for reproducibility.
