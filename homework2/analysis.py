import os
import warnings
import pandas as pd
from stats_functions import *

warnings.filterwarnings("ignore", message="Could not infer format")

######## Helpers


def load_fitbit_daily_steps(filepath):
    """
    Fitbit minuteSteps files are in an 'hour-row' format:
    Steps00..Steps59 represent each minute within that hour.
    So: sum across the 60 minute columns -> hourly steps,
    then sum hours by date -> daily total steps.
    """
    df = pd.read_csv(filepath)
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], errors="coerce")

    step_cols = [c for c in df.columns if c.startswith("Steps")]
    df["HourlySteps"] = df[step_cols].sum(axis=1)

    df["Date"] = df["ActivityHour"].dt.date
    daily = df.groupby("Date")["HourlySteps"].sum()
    return daily


def read_actigraph_start_datetime(filepath):
    """
    ActiGraph week files have a metadata header.
    We pull Start Date + Start Time so we can rebuild timestamps.
    """
    start_date = None
    start_time = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Start Date"):
                start_date = line.replace("Start Date", "").strip()
            elif line.startswith("Start Time"):
                start_time = line.replace("Start Time", "").strip()

            if start_date and start_time:
                break

    if not (start_date and start_time):
        raise ValueError("Couldn't find Start Date/Start Time in ActiGraph header.")

    return pd.to_datetime(f"{start_date} {start_time}", format="%m/%d/%Y %H:%M:%S")


def load_actigraph_daily_steps(filepath, steps_col_index=3):
    """
    After metadata (first 10 lines), ActiGraph file is just numbers (9 columns),
    one row per minute.

    IMPORTANT: There's no real column header, so we use steps_col_index.
    From the preview, col 3 (4th value) looked like steps/min (small integers).
    If your results ever look totally wrong, try a different column index.
    """
    start_dt = read_actigraph_start_datetime(filepath)

    df = pd.read_csv(filepath, skiprows=10, header=None, engine="python", on_bad_lines="skip")

    df["DateTime"] = start_dt + pd.to_timedelta(df.index, unit="min")
    df["Date"] = df["DateTime"].dt.date

    steps = df.iloc[:, steps_col_index]
    daily = steps.groupby(df["Date"]).sum()
    return daily



########### Folder paths


FITBIT_DIR = "data/actigraph and fitbit/fitbit"
ACTIGRAPH_DIR = "data/actigraph and fitbit/actigraph"
MULTIYEAR_DIR = "data/multiyear"



############ Q1: Daily Steps (Fitbit)

all_daily_steps = []

for fname in os.listdir(FITBIT_DIR):
    if fname.endswith("minuteSteps.csv"):
        daily = load_fitbit_daily_steps(os.path.join(FITBIT_DIR, fname))
        all_daily_steps.extend(daily.tolist())

print("\n=== Q1: Daily Steps (Fitbit) ===")
print("Days counted:", len(all_daily_steps))
print("Arithmetic mean:", arithmetic_mean(all_daily_steps))
print("Harmonic mean:", harmonic_mean(all_daily_steps))



############ Q2: Group variance (pooled std dev across subjects)

subject_daily = {}  # subject -> list of daily totals

for fname in os.listdir(FITBIT_DIR):
    if fname.endswith("minuteSteps.csv"):
        subject = fname.split("_")[0]  # "1", "2", etc
        daily = load_fitbit_daily_steps(os.path.join(FITBIT_DIR, fname))
        subject_daily[subject] = daily.tolist()

std_list = []
n_list = []

for subject, days in subject_daily.items():
    std_list.append(standard_deviation(days))
    n_list.append(len(days))

print("\n=== Q2: Group Variance (Fitbit) ===")
print("Subjects:", len(subject_daily))
print("Pooled std dev:", pooled_std(std_list, n_list))


################ Q3: Compare devices (t-test) using Subject 1 overlapping days

fb_daily_1 = load_fitbit_daily_steps(os.path.join(FITBIT_DIR, "1_FB_minuteSteps.csv"))

ag_parts = []
for wk in ["1_AG_week1.csv", "1_AG_week2.csv"]:
    path = os.path.join(ACTIGRAPH_DIR, wk)
    if os.path.exists(path):
        ag_parts.append(load_actigraph_daily_steps(path, steps_col_index=3))

ag_daily_1 = pd.concat(ag_parts).groupby(level=0).sum()

common_days = sorted(set(fb_daily_1.index).intersection(set(ag_daily_1.index)))
fb_vals = [float(fb_daily_1[d]) for d in common_days]
ag_vals = [float(ag_daily_1[d]) for d in common_days]

print("\n=== Q3: Fitbit vs ActiGraph (Subject 1) ===")
print("Overlapping days:", len(common_days))
print("Date range:", common_days[0], "to", common_days[-1])

t, p = t_test(data1=fb_vals, data2=ag_vals)
print("t-value:", t)
print("p-value:", p)

print("(ActiGraph steps_col_index used = 3)")

############# Q4: Weekend warriors (ANOVA by day of week)


dow = {i: [] for i in range(7)}  # 0=Mon ... 6=Sun

for fname in os.listdir(FITBIT_DIR):
    if fname.endswith("minuteSteps.csv"):
        daily = load_fitbit_daily_steps(os.path.join(FITBIT_DIR, fname))
        for d, steps in daily.items():
            day_index = pd.to_datetime(d).dayofweek
            dow[day_index].append(float(steps))

groups = [dow[i] for i in range(7)]

print("\n=== Q4: Day-of-week ANOVA (Fitbit) ===")
print("Counts:", [len(g) for g in groups])

F, p = one_way_anova(*groups)
print("F-value:", F)
print("p-value:", p)


############## Q5: Seasonality (Repeated Measures ANOVA across months)
# Multiyear -> monthly average steps
# Rows = years, cols = months (1..12)

print("\n=== Q5: Seasonality (Multiyear) ===")

steps_file = os.path.join(MULTIYEAR_DIR, "dailySteps.csv")
df = pd.read_csv(steps_file)

df["ActivityDay"] = pd.to_datetime(df["ActivityDay"], errors="coerce")
df = df.dropna(subset=["ActivityDay", "StepTotal"])

df["Year"] = df["ActivityDay"].dt.year
df["Month"] = df["ActivityDay"].dt.month

monthly = df.groupby(["Year", "Month"])["StepTotal"].mean().reset_index()
pivot = monthly.pivot(index="Year", columns="Month", values="StepTotal")

# pick the two years with the most month coverage
month_counts = pivot.notna().sum(axis=1).sort_values(ascending=False)
best_years = list(month_counts.index[:2])
pivot_two = pivot.loc[best_years].copy()

# fill missing months in each year using that year's average (simple fix)
pivot_two = pivot_two.apply(lambda row: row.fillna(row.mean()), axis=1)
pivot_two = pivot_two.dropna()

print("Using years:", list(pivot_two.index))
print("Month counts:\n", month_counts)

data_matrix = []
for yr in pivot_two.index:
    row = [float(pivot_two.loc[yr, m]) for m in range(1, 13)]
    data_matrix.append(row)

F_rm, p_rm = repeated_measures_anova(data_matrix)

print("F-value:", F_rm)
print("p-value:", p_rm)