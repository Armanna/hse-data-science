"""
Merge raw tweet CSVs, clean text, deduplicate, and filter.
Outputs a clean dataset ready for emotion labeling.
"""

import csv
import os
import re
import glob

INPUT_DIR = "output"
OUTPUT_DIR = "processed_data"
MIN_WORDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_tweet_text(text):
    # Normalize unicode quotes/dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b\w+\.\w{2,4}/\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove RT prefix
    text = re.sub(r'^RT\s*:?\s*', '', text, flags=re.IGNORECASE)
    # Hashtags to plain words
    text = re.sub(r'#(\w+)', r'\1', text)
    # Strip leftover special chars
    text = re.sub(r"[^\w\s.,!?'\"-:;()]", ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_english(text):
    latin_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    if latin_alpha < 3:
        return False
    return (latin_alpha / total_alpha) >= 0.70


def has_enough_words(text):
    return len(text.split()) >= MIN_WORDS


def is_promotional(text):
    lower = text.lower()
    promo_patterns = [
        r'\bfollow\b.*\blike\b.*\bpost\b',
        r'\bretweet\b.*\bwin\b',
        r'\bgiveaway\b',
        r'\bfollow\s+(and|&)\s+(retweet|rt)\b',
        r'\blike\s+(and|&)\s+(follow|subscribe)\b',
        r'\buse\s+code\b',
        r'\bdiscount\s+code\b',
        r'\bpromo\s+code\b',
        r'\bsign\s+up\b.*\bfree\b',
        r'\bbet\s+now\b',
        r'\bfree\s+bet\b',
        r'\bodds\s+boost\b',
    ]
    for pattern in promo_patterns:
        if re.search(pattern, lower):
            return True
    return False


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    print(f"Found {len(files)} CSV files to process")

    all_tweets = []
    for f in files:
        basename = os.path.basename(f)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
        source_date = date_match.group(1) if date_match else "unknown"

        with open(f, 'r', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if row:
                    raw_text = row[0].strip()
                    if raw_text:
                        all_tweets.append({
                            "raw_text": raw_text,
                            "source_date": source_date
                        })

    print(f"Total tweets loaded: {len(all_tweets)}")

    for tweet in all_tweets:
        tweet["text"] = clean_tweet_text(tweet["raw_text"])

    non_empty = [t for t in all_tweets if t["text"]]
    print(f"After removing empty (post-clean): {len(non_empty)}")

    # Deduplicate by normalized text
    seen = set()
    unique_tweets = []
    for tweet in non_empty:
        normalized = tweet["text"].lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_tweets.append(tweet)

    duplicates_removed = len(non_empty) - len(unique_tweets)
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Unique tweets: {len(unique_tweets)}")

    long_enough = [t for t in unique_tweets if has_enough_words(t["text"])]
    print(f"Removed (too short): {len(unique_tweets) - len(long_enough)}")

    english_tweets = [t for t in long_enough if is_english(t["text"])]
    print(f"Removed (non-English): {len(long_enough) - len(english_tweets)}")

    clean_tweets = [t for t in english_tweets if not is_promotional(t["text"])]
    print(f"Removed (promotional): {len(english_tweets) - len(clean_tweets)}")

    fieldnames = ["text", "source_date"]

    def write_csv(filepath, data):
        with open(filepath, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        print(f"  Written: {filepath} ({len(data)} rows)")

    write_csv(os.path.join(OUTPUT_DIR, "tweets_clean.csv"), clean_tweets)

    print(f"\nFinal clean dataset: {len(clean_tweets)} tweets")


if __name__ == "__main__":
    main()
