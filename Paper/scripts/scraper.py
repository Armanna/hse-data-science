import argparse
import csv
import json
import os
import re
import time
import random
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

MAX_TWEETS_PER_DATE = 300
MAX_RUNTIME_PER_DATE = 2 * 60
SCROLL_RETRIES = 6

KEYWORDS_LIST = [
    "(UFC OR MMA OR #UFC OR #MMA)",
    "(UFC OR MMA OR #UFCFightNight OR PPV)"
]


def is_english_text(text):
    text = text.strip()
    if not text:
        return False
    return bool(re.search(r"[A-Za-z]", text))


def generate_date_ranges(start, end, step):
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    while start < end:
        next_date = start + timedelta(days=step)
        yield start.date()
        start = next_date


def configure_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument(f"user-agent={USER_AGENT}")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def load_cookies(driver, cookies_path):
    driver.get("https://x.com/")
    time.sleep(6)
    with open(cookies_path, "r", encoding="utf-8") as f:
        cookies = json.load(f)
    for cookie in cookies:
        cookie.pop("sameSite", None)
        cookie.pop("storeId", None)
        driver.add_cookie(cookie)
    driver.refresh()
    time.sleep(5)


def build_search_query(keyword, date_from, date_to):
    return f"{keyword} lang:en min_faves:25 since:{date_from} until:{date_to}"


def scrape_for_date(driver, keyword, date_from, date_to):
    start_time = time.time()
    tweets = []
    retries = SCROLL_RETRIES

    driver.get("https://x.com/explore")
    time.sleep(7)

    search_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "input"))
    )

    search_input.clear()
    search_input.send_keys(build_search_query(keyword, date_from, date_to))
    search_input.send_keys(Keys.ENTER)
    time.sleep(8)

    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        if time.time() - start_time > MAX_RUNTIME_PER_DATE:
            print("Time limit reached")
            break
        if len(tweets) >= MAX_TWEETS_PER_DATE:
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(4)

        tweet_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')

        for tweet in tweet_elements:
            try:
                text = tweet.find_element(
                    By.CSS_SELECTOR, 'div[data-testid="tweetText"]'
                ).text
                if text and is_english_text(text):
                    tweets.append(text)
                if len(tweets) >= MAX_TWEETS_PER_DATE:
                    break
            except (NoSuchElementException, StaleElementReferenceException):
                continue

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            retries -= 1
            if retries == 0:
                print("Scroll limit reached")
                break
            time.sleep(5)
        else:
            last_height = new_height
            retries = SCROLL_RETRIES

    return tweets


def save_results(tweets, keyword, date_from, output_dir):
    safe_keyword = keyword.replace("(", "").replace(")", "").replace(" ", "_").replace("#", "")
    safe_name = f"{safe_keyword}_{date_from}"
    path = os.path.join(output_dir, f"tweets_{safe_name}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for t in tweets:
            writer.writerow([t])


def get_scraped_dates(output_dir):
    scraped = set()
    if os.path.exists(output_dir):
        for fname in os.listdir(output_dir):
            if fname.startswith("tweets_") and fname.endswith(".csv"):
                match = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", fname)
                if match:
                    scraped.add(match.group(1))
    return scraped


def main():
    parser = argparse.ArgumentParser(description="Scrape UFC/MMA tweets from Twitter/X")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-05-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--step-days", type=int, default=1, help="Days per search window")
    parser.add_argument("--output-dir", default="output", help="Output directory for CSVs")
    parser.add_argument("--cookies", required=True, help="Path to cookies.json for Twitter auth")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scraped_dates = get_scraped_dates(args.output_dir)
    total_dates = 0
    skipped = 0

    for date in generate_date_ranges(args.start_date, args.end_date, args.step_days):
        date_from = date
        date_to = date + timedelta(days=args.step_days)

        if str(date_from) in scraped_dates:
            skipped += 1
            continue

        total_dates += 1
        keyword = random.choice(KEYWORDS_LIST)
        print(f"\n[{total_dates}] Collecting {date_from} -> {date_to} | Keyword: {keyword}")

        driver = configure_driver()
        try:
            load_cookies(driver, args.cookies)
            tweets = scrape_for_date(driver, keyword, str(date_from), str(date_to))
            save_results(tweets, keyword, str(date_from), args.output_dir)
            print(f"Saved {len(tweets)} tweets")
        except Exception as e:
            print(f"Error: {e}")
            save_results([], keyword, str(date_from), args.output_dir)
        finally:
            driver.quit()
            time.sleep(random.uniform(8, 15))

    if skipped:
        print(f"\nSkipped {skipped} already-scraped dates (resume)")


if __name__ == "__main__":
    main()
