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


highlight_rules = {
    1: {"threshold": 800},
    2: {"threshold": 900},
    3: {"threshold": 1000},
    4: {"threshold": 1100},
    5: {"threshold": 1200},
    6: {"threshold": 1250},
    7: {"intercept": 750, "slope": 1.0},
    8: {"intercept": 1000, "slope": 0.5},
    9: {"intercept": 1100, "slope": 0.5},
    10: {"intercept": 1200, "slope": 0.5},
    11: {"intercept": 1300, "slope": 0.5},
    12: {"intercept": 1400, "slope": 0.5},
    13: {"intercept": 1500, "slope": 0.5},
    14: {"intercept": 1500, "slope": 0.5},
}

def highlight_points(df):
    """Return subset of df that should be highlighted based on day-specific receipt rules."""
    if df.empty:
        return df
    day = df["trip_duration_days"].iloc[0]
    rule = highlight_rules.get(day)
    if rule is None:
        return df.iloc[0:0]
    if "threshold" in rule:
        cond = df["total_receipts_amount"] > rule["threshold"]
    else:
        cond = df["total_receipts_amount"] > rule["intercept"] + rule["slope"] * df["miles_traveled"]
    return df[cond]


def analyze_day_split(df, day, threshold=None, intercept=None, slope=None):
    """Split data for a given day based on expected output or a line and plot."""
    subset = df[df["trip_duration_days"] == day]
    if subset.empty:
        return

    if threshold is not None:
        cond = subset["expected_output"] <= threshold
        low_label = f"{day}d <={threshold}"
        high_label = f"{day}d >{threshold}"
    else:
        line_vals = intercept + slope * subset["miles_traveled"]
        cond = subset["expected_output"] <= line_vals
        low_label = f"{day}d below {intercept}+{slope}x"
        high_label = f"{day}d above {intercept}+{slope}x"

    splits = [(subset[cond], low_label, "low"), (subset[~cond], high_label, "high")]

    for part_df, label, suffix in splits:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if not part_df.empty:
            model = scatter_with_regression(part_df, "miles_traveled", "expected_output", ax=axes[0], title=f"{label} Miles vs Output")
            sns.scatterplot(data=part_df, x="total_receipts_amount", y="expected_output", ax=axes[1])
            axes[1].set_title(f"{label} Receipts vs Output")
            highlight = highlight_points(part_df)
            if not highlight.empty:
                axes[0].scatter(highlight["miles_traveled"], highlight["expected_output"], color="orange", label="highlight")
                axes[1].scatter(highlight["total_receipts_amount"], highlight["expected_output"], color="orange", label="highlight")
            fig.tight_layout()
            plt.savefig(f"day_{day}_{suffix}.png")
            plt.close(fig)
            if len(part_df) >= 2:
                print(model.summary())
            else:
                print(f"Not enough data for regression {label}")
        else:
            axes[0].set_title(f"No data for {label}")
            axes[1].axis('off')
            fig.tight_layout()
            plt.savefig(f"day_{day}_{suffix}_empty.png")
            plt.close(fig)


def highlight_duration_receipt_groups(df, axes):
    """Overlay duration/receipt-based splits on existing scatter axes."""
    groups = [
        ((1, 5), 1000),
        ((6, 10), 1200),
        ((10, None), 1500),
    ]
    colors = iter(sns.color_palette("tab10", n_colors=6))
    for (start, end), thresh in groups:
        col_le = next(colors)
        col_gt = next(colors)
        if end is None:
            subset = df[df["trip_duration_days"] >= start]
            label_prefix = f"{start}+d"
        else:
            subset = df[(df["trip_duration_days"] >= start) & (df["trip_duration_days"] <= end)]
            label_prefix = f"{start}-{end}d"
        le_df = subset[subset["total_receipts_amount"] <= thresh]
        gt_df = subset[subset["total_receipts_amount"] > thresh]
        axes[0].scatter(le_df["trip_duration_days"], le_df["expected_output"], marker="x", color=col_le, label=f"{label_prefix} <= {thresh}")
        axes[0].scatter(gt_df["trip_duration_days"], gt_df["expected_output"], marker="x", color=col_gt, label=f"{label_prefix} > {thresh}")
        axes[1].scatter(le_df["miles_traveled"], le_df["expected_output"], marker="x", color=col_le)
        axes[1].scatter(gt_df["miles_traveled"], gt_df["expected_output"], marker="x", color=col_gt)
        axes[2].scatter(le_df["total_receipts_amount"], le_df["expected_output"], marker="x", color=col_le)
        axes[2].scatter(gt_df["total_receipts_amount"], gt_df["expected_output"], marker="x", color=col_gt)


def analyze_duration_receipt_groups(df):
    """Create scatter/regression plots for duration/receipt-based splits."""
    configs = [
        ((1, 5), 1000),
        ((6, 10), 1200),
        ((10, None), 1500),
    ]
    for (start, end), thresh in configs:
        if end is None:
            range_df = df[df["trip_duration_days"] >= start]
            label_prefix = f"{start}+d"
        else:
            range_df = df[(df["trip_duration_days"] >= start) & (df["trip_duration_days"] <= end)]
            label_prefix = f"{start}-{end}d"
        df_le = range_df[range_df["total_receipts_amount"] <= thresh]
        df_gt = range_df[range_df["total_receipts_amount"] > thresh]
        for part_df, tag in [(df_le, "le"), (df_gt, "gt")]:
            suffix = f"{label_prefix}_{tag}{thresh}"
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            if not part_df.empty:
                model = scatter_with_regression(part_df, "miles_traveled", "expected_output", ax=axes[0], title=f"{suffix} Miles vs Output")
                sns.scatterplot(data=part_df, x="total_receipts_amount", y="expected_output", ax=axes[1])
                axes[1].set_title(f"{suffix} Receipts vs Output")
                highlight = highlight_points(part_df)
                if not highlight.empty:
                    axes[0].scatter(highlight["miles_traveled"], highlight["expected_output"], color="orange", label="highlight")
                    axes[1].scatter(highlight["total_receipts_amount"], highlight["expected_output"], color="orange", label="highlight")
                fig.tight_layout()
                plt.savefig(f"duration_split_{suffix}.png")
                plt.close(fig)
                if len(part_df) >= 2:
                    print(model.summary())
                else:
                    print(f"Not enough data for regression {suffix}")
            else:
                axes[0].set_title("No data")
                axes[1].axis('off')
                fig.tight_layout()
                plt.savefig(f"duration_split_{suffix}_empty.png")
                plt.close(fig)


def output_day_tables(df, thresholds):
    """Save CSV tables for day-based expected output splits."""
    for day, thresh in thresholds.items():
        day_df = df[df["trip_duration_days"] == day]
        if day_df.empty:
            continue
        low = day_df[day_df["expected_output"] <= thresh]
        high = day_df[day_df["expected_output"] > thresh]
        low.to_csv(f"day_{day}_le_{thresh}.csv", index=False)
        high.to_csv(f"day_{day}_gt_{thresh}.csv", index=False)
        print(f"Day {day} <= {thresh}: {len(low)} records")
        print(f"Day {day} > {thresh}: {len(high)} records")

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
    highlight_duration_receipt_groups(df, axes)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    plt.savefig("basic_scatter.png")
    plt.close(fig)

    # Scatter plots of trip duration against other inputs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=df, x="trip_duration_days", y="miles_traveled", ax=axes[0])
    axes[0].set_title("Trip Days vs Miles")
    sns.scatterplot(data=df, x="trip_duration_days", y="total_receipts_amount", ax=axes[1])
    axes[1].set_title("Trip Days vs Receipts")
    fig.tight_layout()
    plt.savefig("duration_vs_inputs.png")
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
    analyze_duration_receipt_groups(df)

    # New receipt split at $1500 to check for possible penalties
    df_low_1500 = df[df["total_receipts_amount"] <= 1500]
    df_high_1500 = df[df["total_receipts_amount"] > 1500]

    fig, ax = plt.subplots()
    model_low_1500 = scatter_with_regression(
        df_low_1500,
        "total_receipts_amount",
        "expected_output",
        ax=ax,
        title="Receipts <=1500",
    )
    plt.savefig("receipts_low1500_regression.png")
    plt.close(fig)
    print(model_low_1500.summary())

    fig, ax = plt.subplots()
    model_high_1500 = scatter_with_regression(
        df_high_1500,
        "total_receipts_amount",
        "expected_output",
        ax=ax,
        title="Receipts >1500",
    )
    plt.savefig("receipts_high1500_regression.png")
    plt.close(fig)
    print(model_high_1500.summary())

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
            axes[0].scatter(highlight["miles_traveled"], highlight["expected_output"], color="orange", label="highlight")
            axes[1].scatter(highlight["total_receipts_amount"], highlight["expected_output"], color="orange", label="highlight")
        fig.tight_layout()
        plt.savefig(f"day_{day}_analysis.png")
        plt.close(fig)

    # Further splits for each day based on expected output thresholds or lines
    day_params = {
        1: {"threshold": 800},
        2: {"threshold": 900},
        3: {"threshold": 1000},
        4: {"threshold": 1100},
        5: {"threshold": 1200},
        6: {"threshold": 1250},
        7: {"intercept": 750, "slope": 1.0},
        8: {"intercept": 1000, "slope": 0.5},
        9: {"intercept": 1100, "slope": 0.5},
        10: {"intercept": 1200, "slope": 0.5},
        11: {"intercept": 1300, "slope": 0.5},
        12: {"intercept": 1400, "slope": 0.5},
        13: {"intercept": 1500, "slope": 0.5},
        14: {"intercept": 1500, "slope": 0.5},
    }

    for day, params in day_params.items():
        analyze_day_split(df, day, **params)

    # Output CSV tables for the first three days to inspect split models
    output_day_tables(df, {1: 800, 2: 900, 3: 1000})

    # Additional simple analysis: correlation matrix
    corr = df.corr(numeric_only=True)
    print("Correlation matrix:\n", corr)
    corr.to_csv("correlations.csv")

if __name__ == "__main__":
    analyze()
