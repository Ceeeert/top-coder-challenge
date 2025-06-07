import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit


def load_data(path="public_cases.json"):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for rec in data:
        inp = rec["input"]
        row = {
            "trip_duration_days": inp["trip_duration_days"],
            "miles_traveled": inp["miles_traveled"],
            "total_receipts_amount": inp["total_receipts_amount"],
            "expected_output": rec["expected_output"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def scatter_with_regression(df, x, y, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    # Fit linear regression
    model = smf.ols(f"{y} ~ {x}", data=df).fit()
    xs = np.linspace(df[x].min(), df[x].max(), 100)
    ys = model.predict(pd.DataFrame({x: xs}))
    ax.plot(xs, ys, color="red")
    if len(model.params) > 1:
        intercept = model.params.get("Intercept", model.params.iloc[0])
        slope = model.params.get(x, model.params.iloc[1])
        eq = f"y={slope:.2f}x+{intercept:.2f}"
    else:
        eq = ""
    if title:
        ax.set_title(title + "\n" + eq)
    else:
        ax.set_title(eq)
    return model


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def logistic_fit(df, xcol, ycol):
    x = df[xcol].values
    y = df[ycol].values
    p0 = [y.max(), 0.01, x.mean()]
    params, cov = curve_fit(logistic, x, y, p0=p0, maxfev=10000)
    residuals = y - logistic(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    return params, r2


def highlight_points(df):
    decimals = (df["total_receipts_amount"] % 1).round(2)
    pattern_points = df[(decimals.isin([0.49, 0.99])) | ((df["total_receipts_amount"] >= 840) & (df["total_receipts_amount"] <= 860))]
    return pattern_points

def check_duplicates(df):
    # identical across all three inputs
    dup_all = df[df.duplicated(["trip_duration_days", "miles_traveled", "total_receipts_amount"], keep=False)]
    if dup_all.empty:
        print("No exact duplicate input records found.")
    else:
        print("Duplicate records with all inputs equal:")
        for key, grp in dup_all.groupby(["trip_duration_days", "miles_traveled", "total_receipts_amount"]):
            print("Inputs:", key)
            print(grp[["expected_output"]])
            if grp["expected_output"].nunique() > 1:
                print("Outputs differ!")
            else:
                print("Outputs identical.")

    # duplicates on any pair of inputs
    pairs = [
        ("trip_duration_days", "miles_traveled"),
        ("trip_duration_days", "total_receipts_amount"),
        ("miles_traveled", "total_receipts_amount"),
    ]
    for cols in pairs:
        dup = df[df.duplicated(list(cols), keep=False)]
        if dup.empty:
            continue
        print(f"\nDuplicate records on {cols}:")
        for key, grp in dup.groupby(list(cols)):
            if len(grp) > 1:
                print("Inputs:", key)
                print(grp[["expected_output"]])
                if grp["expected_output"].nunique() > 1:
                    print("Outputs differ!")
                else:
                    print("Outputs identical.")


def analyze():
    df = load_data()
    print(df.head())
    check_duplicates(df)

    # Basic scatter plots
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.scatterplot(data=df, x="trip_duration_days", y="expected_output", ax=axes[0])
    axes[0].set_title("Trip Duration vs Output")
    sns.scatterplot(data=df, x="miles_traveled", y="expected_output", ax=axes[1])
    axes[1].set_title("Miles vs Output")
    sns.scatterplot(data=df, x="total_receipts_amount", y="expected_output", ax=axes[2])
    axes[2].set_title("Receipts vs Output")
    fig.tight_layout()
    plt.savefig("basic_scatter.png")
    plt.close(fig)

    # Linear regression: miles_traveled vs expected_output
    fig, ax = plt.subplots()
    model_miles = scatter_with_regression(df, "miles_traveled", "expected_output", ax=ax, title="Miles vs Output")
    plt.savefig("miles_regression.png")
    plt.close(fig)
    print(model_miles.summary())

    # Receipt subsets
    df_low = df[df["total_receipts_amount"] <= 1200]
    df_high = df[df["total_receipts_amount"] > 1200]

    fig, ax = plt.subplots()
    model_low = scatter_with_regression(df_low, "total_receipts_amount", "expected_output", ax=ax, title="Receipts <=1200")

    plt.savefig("receipts_low_regression.png")
    plt.close(fig)
    print(model_low.summary())

    fig, ax = plt.subplots()
    model_high = scatter_with_regression(df_high, "total_receipts_amount", "expected_output", ax=ax, title="Receipts >1200")
    plt.savefig("receipts_high_regression.png")
    plt.close(fig)
    print(model_high.summary())

    params, r2 = logistic_fit(df, "total_receipts_amount", "expected_output")
    print("Logistic fit params (L,k,x0):", params)
    print("Logistic fit R^2:", r2)
    xline = np.linspace(df["total_receipts_amount"].min(), df["total_receipts_amount"].max(), 200)
    yline = logistic(xline, *params)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="total_receipts_amount", y="expected_output", ax=ax)
    ax.plot(xline, yline, color="red")
    ax.set_title("Logistic Receipts vs Output")
    plt.savefig("logistic_receipts.png")
    plt.close(fig)

    # Trip duration subsets
    df_short = df[df["trip_duration_days"] <= 7]
    df_long = df[df["trip_duration_days"] > 7]

    fig, ax = plt.subplots()
    model_short = scatter_with_regression(df_short, "trip_duration_days", "expected_output", ax=ax, title="Trip Days <=7")

    plt.savefig("trip_short_regression.png")
    plt.close(fig)
    print(model_short.summary())

    fig, ax = plt.subplots()
    model_long = scatter_with_regression(df_long, "trip_duration_days", "expected_output", ax=ax, title="Trip Days >7")
    plt.savefig("trip_long_regression.png")
    plt.close(fig)
    print(model_long.summary())

    # Scatter plots by trip_duration_days
    unique_days = sorted(df["trip_duration_days"].unique())
    for day in unique_days:
        subset = df[df["trip_duration_days"] == day]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        model = scatter_with_regression(subset, "miles_traveled", "expected_output", ax=axes[0], title=f"{day} days: Miles vs Output")
        sns.scatterplot(data=subset, x="total_receipts_amount", y="expected_output", ax=axes[1])
        axes[1].set_title(f"{day} days: Receipts vs Output")
        highlight = highlight_points(subset)
        if not highlight.empty:
            axes[0].scatter(highlight["miles_traveled"], highlight["expected_output"], color="orange", label="pattern")
            axes[1].scatter(highlight["total_receipts_amount"], highlight["expected_output"], color="orange", label="pattern")
            axes[0].legend()
            axes[1].legend()
        fig.tight_layout()
        plt.savefig(f"day_{day}_analysis.png")
        plt.close(fig)

    # Additional simple analysis: correlation matrix
    corr = df.corr(numeric_only=True)
    print("Correlation matrix:\n", corr)
    corr.to_csv("correlations.csv")

if __name__ == "__main__":
    analyze()
