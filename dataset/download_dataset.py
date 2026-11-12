"""
Download the SMS Spam Collection dataset (UCI / public domain).
Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
Mirror: justmarkham/pycon-2016-tutorial (same data, TSV format)
"""
import urllib.request
import os
import csv

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
OUT = os.path.join(os.path.dirname(__file__), "spam.csv")


def download():
    print("Downloading SMS Spam Collection dataset...")
    with urllib.request.urlopen(URL) as resp:
        raw = resp.read().decode("utf-8")

    lines = raw.splitlines()
    rows = []
    for line in lines:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            label, msg = parts
            label = label.strip().lower()
            msg = msg.strip()
            if label in ("spam", "ham") and msg:
                rows.append({"label": label, "message": msg})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "message"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUT}")
    spam_count = sum(1 for r in rows if r["label"] == "spam")
    ham_count = sum(1 for r in rows if r["label"] == "ham")
    print(f"  Spam: {spam_count}  Ham: {ham_count}")


if __name__ == "__main__":
    download()
