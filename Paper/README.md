# A Deep Learning Approach to Automatically Identify Emotions in Text Data

Transformer-based emotion classification on a dataset of ~96k English-language tweets related to UFC and MMA events, collected from Twitter/X (January 2020 -- May 2026). Tweets are labeled with seven emotion categories (Ekman's six basic emotions + neutral). We fine-tune BERT and RoBERTa with focal loss and compare against TF-IDF baselines (Logistic Regression, SVM, Naive Bayes).

## Repository Structure

```
scripts/          Data collection and processing pipeline
  scraper.py          Tweet collection via Selenium
  process_tweets.py   Text cleaning, deduplication, filtering
  label_emotions.py   Automated emotion labeling (distilroberta)
  compute_validation.py   Inter-annotator agreement metrics
notebooks/
  experiments.ipynb   Model training and evaluation
data/
  tweets_labeled.csv          Labeled dataset (96,491 tweets)
  manual_validation_200.csv   200-tweet validation sample with human annotations
figures/          All paper figures
```

## Reproducing Results

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Data collection (optional -- labeled data already provided in `data/`):
   ```
   python scripts/scraper.py --cookies path/to/cookies.json
   python scripts/process_tweets.py
   python scripts/label_emotions.py
   ```

3. Run experiments:
   Open `notebooks/experiments.ipynb` and run all cells. Training uses GPU if available, otherwise falls back to CPU.

## Dataset

- **Source:** Twitter/X, keyword search (UFC, MMA, #UFC, #MMA, #UFCFightNight, PPV)
- **Period:** January 2020 -- May 2026
- **Size:** 96,491 unique tweets after deduplication and filtering
- **Labels:** neutral, joy, anger, sadness, surprise, fear, disgust
- **Labeling model:** `j-hartmann/emotion-english-distilroberta-base`
- **Validation:** 200-tweet sample annotated by two human annotators (Cohen's kappa = 0.74)
