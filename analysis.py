import os
import random
import warnings
import pandas as pd
from ml_functions import kmeans, knn, cpa

warnings.filterwarnings("ignore", message="Could not infer format")

FITBIT_DIR = "data/actigraph and fitbit/fitbit"
MULTIYEAR_DIR = "data/multiyear"


def load_fitbit_daily_features(subject_id):
    """
    Build daily Fitbit feature vectors for one subject.
    Each day becomes:
    [steps, calories, intensity, METs]
    """
    steps_file = os.path.join(FITBIT_DIR, f"{subject_id}_FB_minuteSteps.csv")
    calories_file = os.path.join(FITBIT_DIR, f"{subject_id}_FB_minuteCalories.csv")
    intensity_file = os.path.join(FITBIT_DIR, f"{subject_id}_FB_minuteIntensities.csv")
    mets_file = os.path.join(FITBIT_DIR, f"{subject_id}_FB_minuteMETs.csv")

    steps_df = pd.read_csv(steps_file)
    calories_df = pd.read_csv(calories_file)
    intensity_df = pd.read_csv(intensity_file)
    mets_df = pd.read_csv(mets_file)

    steps_df["ActivityHour"] = pd.to_datetime(steps_df["ActivityHour"], errors="coerce")
    calories_df["ActivityHour"] = pd.to_datetime(calories_df["ActivityHour"], errors="coerce")
    intensity_df["ActivityHour"] = pd.to_datetime(intensity_df["ActivityHour"], errors="coerce")
    mets_df["ActivityHour"] = pd.to_datetime(mets_df["ActivityHour"], errors="coerce")

    step_cols = [c for c in steps_df.columns if c.startswith("Steps")]
    cal_cols = [c for c in calories_df.columns if c.startswith("Calories")]
    int_cols = [c for c in intensity_df.columns if c.startswith("Intensity")]
    met_cols = [c for c in mets_df.columns if c.startswith("MET")]

    steps_df["HourlySteps"] = steps_df[step_cols].sum(axis=1)
    calories_df["HourlyCalories"] = calories_df[cal_cols].sum(axis=1)
    intensity_df["HourlyIntensity"] = intensity_df[int_cols].sum(axis=1)
    mets_df["HourlyMETs"] = mets_df[met_cols].sum(axis=1)

    steps_df["Date"] = steps_df["ActivityHour"].dt.date
    calories_df["Date"] = calories_df["ActivityHour"].dt.date
    intensity_df["Date"] = intensity_df["ActivityHour"].dt.date
    mets_df["Date"] = mets_df["ActivityHour"].dt.date

    daily_steps = steps_df.groupby("Date")["HourlySteps"].sum()
    daily_calories = calories_df.groupby("Date")["HourlyCalories"].sum()
    daily_intensity = intensity_df.groupby("Date")["HourlyIntensity"].sum()
    daily_mets = mets_df.groupby("Date")["HourlyMETs"].sum()

    daily_features = pd.DataFrame({
        "steps": daily_steps,
        "calories": daily_calories,
        "intensity": daily_intensity,
        "mets": daily_mets
    }).dropna()

    return daily_features


# build one big dataset of daily features and labels for all subjects
all_features = []
all_labels = []

for subject_id in [1, 2, 3, 4]:
    daily_features = load_fitbit_daily_features(subject_id)

    for row in daily_features.values.tolist():
        all_features.append(row)
        all_labels.append(subject_id)

print("Total feature rows:", len(all_features))
print("Total labels:", len(all_labels))


# group similar daily behavior patterns into clusters
print("\n--- KMEANS RESULTS ---")

k = 4
centroids, cluster_labels = kmeans(all_features, k)

print("Number of centroids:", len(centroids))
print("First centroid:", centroids[0])
print("Cluster counts:", {c: cluster_labels.count(c) for c in set(cluster_labels)})


print("\n--- KNN RESULTS ---")

# shuffle data so training and test sets contain mixed subjects
combined = list(zip(all_features, all_labels))
random.shuffle(combined)

all_features_shuffled, all_labels_shuffled = zip(*combined)

split_index = int(0.8 * len(all_features_shuffled))

X_train = list(all_features_shuffled[:split_index])
y_train = list(all_labels_shuffled[:split_index])

X_test = list(all_features_shuffled[split_index:])
y_test = list(all_labels_shuffled[split_index:])

correct = 0

# predict each test point and compare to true label
for i in range(len(X_test)):
    prediction = knn(X_train, y_train, X_test[i], k=5)
    if prediction == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print("Test samples:", len(X_test))
print("Correct predictions:", correct)
print("Accuracy:", accuracy)


print("\n--- CPA RESULTS ---")

steps_file = os.path.join(MULTIYEAR_DIR, "dailySteps.csv")
steps_df = pd.read_csv(steps_file)

steps_df["ActivityDay"] = pd.to_datetime(steps_df["ActivityDay"], errors="coerce")
steps_df = steps_df.dropna(subset=["ActivityDay", "StepTotal"])
steps_df = steps_df[steps_df["StepTotal"] > 0].reset_index(drop=True)

step_series = steps_df["StepTotal"].tolist()

# detect points where long-term activity behavior changes
change_points = cpa(step_series, max_changes=8)

print("Number of change points found:", len(change_points))
print("Change point indices:", change_points)
print("Change point dates:")
for cp in change_points:
    print(cp, "->", steps_df.loc[cp, "ActivityDay"].date())