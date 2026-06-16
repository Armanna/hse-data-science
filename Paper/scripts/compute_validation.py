"""
Compute inter-annotator agreement and automated label validation metrics.
Expects a CSV with columns: automated_label, annotator_1, annotator_2.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def main():
    df = pd.read_csv("data/manual_validation_200.csv")

    if df['annotator_1'].isna().any() or (df['annotator_1'] == '').any():
        missing = df['annotator_1'].isna().sum() + (df['annotator_1'] == '').sum()
        print(f"ERROR: {missing} rows missing in annotator_1 column.")
        return
    if df['annotator_2'].isna().any() or (df['annotator_2'] == '').any():
        missing = df['annotator_2'].isna().sum() + (df['annotator_2'] == '').sum()
        print(f"ERROR: {missing} rows missing in annotator_2 column.")
        return

    df['annotator_1'] = df['annotator_1'].str.strip().str.lower()
    df['annotator_2'] = df['annotator_2'].str.strip().str.lower()
    df['automated_label'] = df['automated_label'].str.strip().str.lower()

    valid_labels = {'neutral', 'joy', 'anger', 'sadness', 'surprise', 'disgust', 'fear'}

    for col in ['annotator_1', 'annotator_2']:
        invalid = set(df[col].unique()) - valid_labels
        if invalid:
            print(f"ERROR: Invalid labels in {col}: {invalid}")
            return

    # Inter-annotator agreement
    kappa = cohen_kappa_score(df['annotator_1'], df['annotator_2'])
    annotator_agreement = (df['annotator_1'] == df['annotator_2']).mean()

    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 60)
    print(f"Cohen's kappa:    {kappa:.2f}")
    print(f"Raw agreement:    {annotator_agreement:.1%}")
    print()

    # Consensus labels (agreement -> use shared label, else use annotator_1)
    df['consensus'] = None
    agree_mask = df['annotator_1'] == df['annotator_2']
    df.loc[agree_mask, 'consensus'] = df.loc[agree_mask, 'annotator_1']

    n_disagree = (~agree_mask).sum()
    if n_disagree > 0:
        print(f"NOTE: {n_disagree} tweets where annotators disagree.")
        print("Using annotator_1's label as tiebreaker.")
        df.loc[~agree_mask, 'consensus'] = df.loc[~agree_mask, 'annotator_1']
    print()

    # Automated vs consensus
    overall_agreement = (df['automated_label'] == df['consensus']).mean()

    print("=" * 60)
    print("AUTOMATED LABELS vs. HUMAN CONSENSUS")
    print("=" * 60)
    print(f"Overall agreement: {overall_agreement:.1%}")
    print()

    emotions = ['neutral', 'joy', 'anger', 'sadness', 'surprise', 'disgust', 'fear']
    print("Per-class accuracy (automated vs. consensus):")
    for emotion in emotions:
        mask = df['consensus'] == emotion
        if mask.sum() > 0:
            acc = (df.loc[mask, 'automated_label'] == emotion).mean()
            print(f"  {emotion:12s}: {acc:.1%}  (n={mask.sum()})")
        else:
            print(f"  {emotion:12s}: N/A  (n=0)")
    print()

    # Confusion matrix
    print("=" * 60)
    print("CONFUSION MATRIX (rows=consensus, cols=automated)")
    print("=" * 60)
    cm = confusion_matrix(df['consensus'], df['automated_label'], labels=emotions)
    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
    print(cm_df)


if __name__ == "__main__":
    main()
