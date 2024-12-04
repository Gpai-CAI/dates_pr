import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# --- Step 1: Generate Synthetic Data ---
np.random.seed(42)  # For reproducibility

# Parameters
num_rows = 20
entity_types = ["medication_start_date", "medication_end_date", "diagnosis_date"]

# Create synthetic ground truth dates
df = pd.DataFrame({
    "entity_type": np.random.choice(entity_types, size=num_rows),
    "date_gt": pd.date_range(start="2024-01-01", periods=num_rows, freq="D")
})

# Introduce random errors for predictions
error_range = np.random.choice([1, 10, 50], size=num_rows, p=[0.7, 0.2, 0.1])  # 70% small errors
error_sign = np.random.choice([-1, 1], size=num_rows)
df["date_pred"] = df["date_gt"] + pd.to_timedelta(error_range * error_sign, unit="D")

# --- Step 2: Calculate Metrics ---
# Exact matches
df["exact_match"] = df["date_gt"] == df["date_pred"]

# Difference in days
df["day_difference"] = (df["date_gt"] - df["date_pred"]).dt.days

# Overall Metrics
metrics_summary = df["day_difference"].describe()

# Matches within ±7 and ±14 days
df["within_7_days"] = df["day_difference"].abs() <= 7
df["within_14_days"] = df["day_difference"].abs() <= 14
within_7_days_accuracy = df["within_7_days"].mean() * 100
within_14_days_accuracy = df["within_14_days"].mean() * 100

# Grouped Metrics
# Grouped Metrics
grouped_metrics = df.groupby("entity_type")[["within_7_days", "within_14_days"]].mean() * 100

# --- Step 3: Visualize Metrics ---
# Box plot for day difference by entity type
plt.figure(figsize=(10, 6))
sns.boxplot(x="entity_type", y="day_difference", data=df, palette="Blues")

# Overlay scatter plot
sns.stripplot(x="entity_type", y="day_difference", data=df, color="red", alpha=0.5, jitter=True)

# Highlight exact match (y=0) with a dashed red line
plt.axhline(0, color="red", linestyle="--", label="Exact Match")

# Adjust y-axis to zoom in on the relevant range
plt.ylim(-10, 10)  # Adjust as needed

# Add labels and title
plt.title("Day Difference Distribution by Entity Type")
plt.xlabel("Entity Type")
plt.ylabel("Day Difference")
plt.legend()

plt.show()


# Histogram of day differences
plt.figure(figsize=(10, 6))
sns.histplot(df["day_difference"], bins=30, kde=True)
plt.title("Distribution of Day Differences")
plt.xlabel("Day Difference")
plt.ylabel("Frequency")
plt.show()

# --- Step 4: Output Metrics ---
print("Overall Metrics Summary:")
print(metrics_summary)

print(f"\nWithin ±7 Days Accuracy: {within_7_days_accuracy:.2f}%")
print(f"Within ±14 Days Accuracy: {within_14_days_accuracy:.2f}%")

print("\nGrouped Metrics (By Entity Type):")
print(grouped_metrics)
