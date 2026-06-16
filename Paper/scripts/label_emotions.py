"""
Label tweets with emotions using a pre-trained classifier.
Uses j-hartmann/emotion-english-distilroberta-base (Ekman 6 + neutral).
"""

import csv
import os
from collections import Counter

from transformers import pipeline

INPUT_FILE = "processed_data/tweets_clean.csv"
OUTPUT_DIR = "processed_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tweets_labeled.csv")
BATCH_SIZE = 32


def load_tweets(filepath):
    tweets = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets.append(row)
    return tweets


def label_tweets(tweets, classifier):
    texts = [t["text"] for t in tweets]
    total = len(texts)
    labels = []

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        results = classifier(batch, truncation=True, max_length=512)
        for result in results:
            if isinstance(result, list):
                labels.append(result[0]["label"])
            else:
                labels.append(result["label"])

        done = min(i + BATCH_SIZE, total)
        if done % 128 < BATCH_SIZE or done == total:
            print(f"  Labeled {done}/{total} tweets...")

    return labels


def main():
    print("Loading emotion classifier...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=-1
    )

    print(f"Loading tweets from {INPUT_FILE}...")
    tweets = load_tweets(INPUT_FILE)
    print(f"Loaded {len(tweets)} tweets")

    print("Labeling tweets...")
    labels = label_tweets(tweets, classifier)

    for tweet, label in zip(tweets, labels):
        tweet["emotion"] = label

    fieldnames = ["text", "source_date", "emotion"]
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tweets)
    print(f"\nLabeled dataset written to: {OUTPUT_FILE}")

    # Print distribution
    emotion_counts = Counter(labels)
    total = len(labels)
    print("\nEmotion distribution:")
    for emotion in ["neutral", "joy", "surprise", "anger", "sadness", "fear", "disgust"]:
        count = emotion_counts.get(emotion, 0)
        pct = (count / total) * 100
        print(f"  {emotion:<12} {count:>6}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<12} {total:>6}")


if __name__ == "__main__":
    main()
