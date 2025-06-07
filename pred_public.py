import json
import pandas as pd
import numpy as np


def load_cases(path="public_cases.json"):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for rec in data:
        inp = rec["input"]
        rows.append({
            "trip_duration_days": inp["trip_duration_days"],
            "miles_traveled": inp["miles_traveled"],
            "total_receipts_amount": inp["total_receipts_amount"],
            "expected_output": rec["expected_output"],
        })
    return pd.DataFrame(rows)


def base_amount(days):
    base = 0
    if days >= 1:
        base += 100
    if days >= 2:
        base += 100
    if days >= 3:
        base += 100
    if days >= 5:
        base += 100
    if days >= 7:
        base += 100
    if days >= 9:
        base += 100
    if days >= 10:
        base += 100
    return base


rate_rules = {
    1: (1000, 0.6, 0.3),
    2: (1000, 0.5, 0.25),
    3: (1000, 0.4, 0.2),
    4: (1000, 0.4, 0.2),
    5: (1000, 0.4, 0.2),
    6: (1200, 0.6, 0.3),
    7: (1200, 1.0, 0.6),
    8: (1200, 0.6, 0.45),
    9: (1200, 0.6, 0.45),
    10: (1200, 0.3, 0.4),
    11: (1500, 0.5, 0.45),
    12: (1500, 0.7, 0.4),
    13: (1500, 0.7, 0.45),
    14: (1500, 0.6, 0.4),
}


def mileage_rate(days, receipts):
    rule = rate_rules.get(days)
    if rule is None:
        # default: use closest lower rule if days>14 else first
        if days > 14:
            rule = rate_rules[14]
        else:
            rule = rate_rules[1]
    thresh, low_rate, high_rate = rule
    return low_rate if receipts <= thresh else high_rate


def predict(row):
    rate = mileage_rate(row["trip_duration_days"], row["total_receipts_amount"])
    base = base_amount(row["trip_duration_days"])
    return row["miles_traveled"] * rate + base


def main():
    df = load_cases()
    df["predicted"] = df.apply(predict, axis=1)
    df["diff"] = np.abs(df["predicted"] - df["expected_output"])
    exact = (df["diff"] <= 0.01).sum()
    close = (df["diff"] <= 1.0).sum()
    avg_dist = df["diff"].mean()
    total_dist = df["diff"].sum()
    print(f"Exact matches (<=0.01): {exact}")
    print(f"Close matches (<=1.00): {close}")
    print(f"Average difference: {avg_dist:.2f}")
    print(f"Total difference: {total_dist:.2f}")


if __name__ == "__main__":
    main()
