import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Task 1
print("TASK 1")
df = pd.read_csv("bike_sharing_hour.csv")

print("First 5 rows:\n", df.head())
print("\nLast 5 rows:", df.tail())
print("Shape:", df.shape)
print("\nColumn names:", df.columns)
print("\nData Types:", df.dtypes)
print("\nData frame info:")
print(df.info())
print("\nMissing values:")
print(df.isna().sum())
print("\nExact duplicates:", df.duplicated().sum())


# Task 2
print("TASK 2")
raw_df = df.copy()
clean_df = df.copy()

output_folder = Path("output")
output_folder.mkdir(parents=True, exist_ok=True)

clean_df = clean_df.rename(columns={
    "dteday": "date",
    "yr": "year_code",
    "mnth": "month",
    "hr": "hour",
    "weathersit": "weather_code",
    "temp": "temperature_normalized",
    "atemp": "apparent_temperature_normalized",
    "hum": "humidity_normalized",
    "windspeed": "wind_speed_normalized",
    "casual": "casual_rentals",
    "registered": "registered_rentals",
    "cnt": "total_rentals"
})
clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
clean_df["year"] = clean_df["year_code"].map({
    0: 2011,
    1: 2012
})

season_map = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}
clean_df["season_label"] = (clean_df["season"].map(season_map))

weather_map = {
    1: "Clear/Partly Cloudy",
    2: "Mist/Cloudy",
    3: "Light Rain/Snow",
    4: "Heavy Rain/Snow/Fog"
}
clean_df["weather_label"] = clean_df["weather_code"].map(weather_map)

day_map = {
    1: "Working Day",
    0: "Nonworking Day"
}
clean_df["day_type"] = clean_df["workingday"].map(day_map)

clean_df["temperature_c"] = (47 * clean_df["temperature_normalized"] - 8)
clean_df["apparent_temperature_c"] = 66 * clean_df["apparent_temperature_normalized"] - 16
clean_df["humidity_percent"] = 100 * clean_df["humidity_normalized"]
clean_df["wind_speed"] = 67 * clean_df["wind_speed_normalized"]

print("Column names:")
print(clean_df.columns)

print("\nData types:")
print(clean_df.dtypes)

print("\nFirst 5 prepared rows:")
print(clean_df[[
    "date", "year", "season_label", "weather_label", "day_type", "temperature_c", "apparent_temperature_c", "humidity_percent", "wind_speed"
]].head())

# Task 3
print("TASK 3")
# 3 - 1
assert clean_df["instant"].is_unique

# 3 - 2
assert clean_df["hour"].between(0, 23).all()

# 3 - 3
assert clean_df["month"].between(1,12).all()

# 3 - 4
assert set(clean_df["season"].unique()).issubset({1, 2, 3, 4})
assert set(clean_df["weather_code"].unique()).issubset({1, 2, 3, 4})

# 3 - 5
assert set(clean_df["holiday"].unique()).issubset({0, 1})
assert set(clean_df["workingday"].unique()).issubset({0, 1})

# 3 - 6
assert (clean_df[["casual_rentals", "registered_rentals", "total_rentals"]] >= 0).all().all()

# 3 - 7
assert (clean_df["total_rentals"] == clean_df["casual_rentals"] + clean_df["registered_rentals"]).all()

# 3 - 8
required_columns = [
    "date", "hour", "season_label", "weather_label",
    "temperature_c", "humidity_percent",
    "wind_speed", "total_rentals"
]
assert clean_df[required_columns].notna().all().all()

# 3 - 9
assert clean_df.duplicated().sum() == 0

# Validation confirmation
print("\nTask 3: Validation Results")
print("All 9 validation checks passed successfully!")


# Task 4
print("TASK 4 A")
# 4A - 1
print("\nTotal Rental Statistics:")
print(clean_df["total_rentals"].agg(["mean", "median", "std", "min", "max"]).round(2))

print("\nTotal Rental Quartiles:")
print(clean_df["total_rentals"].quantile([0.25, 0.5, 0.75]).round(2))

# 4A - 2
print("\nObservations by Season:")
print(clean_df["season_label"].value_counts())

print("\nObservations by Weather:")
print(clean_df["weather_label"].value_counts())

# 4A - 3
print("\nRentals by Day type:")
print(clean_df.groupby("day_type")["total_rentals"].agg(["mean", "median"]).round(2))

# 4A - 4
print("\nMean Rentals by hour:")
print(clean_df.groupby("hour")["total_rentals"].mean().round(2))

#4a - 5
print("\nRentals by Season and Working Dy:")
print(clean_df.groupby(["season_label", "workingday"])["total_rentals"].agg(["count", "mean"]).round(2))

print("TASK 4 B")
# 4B - 1
plt.figure(figsize = (8,5))

sns.histplot(data = clean_df, x = "total_rentals", bins = 30)

plt.title("Distribution of Hourly Bike Rentals")
plt.xlabel("Total Rentals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(output_folder / "rental_histogram.png", dpi = 300)
plt.close()

# 4B - 2
plt.figure(figsize = (8,5))

sns.boxplot(
    data = clean_df,
    x = "season_label",
    y = "total_rentals",
    order = ["Spring", "Summer", "Fall", "Winter"]
)

plt.title("Hourly Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Total Rentals")

plt.tight_layout()
plt.savefig(output_folder / "season_boxplot.png", dpi = 300)
plt.close()

# 4B - 3
hourly_summary = clean_df.groupby(["hour", "day_type"])["total_rentals"].mean().reset_index()

plt.figure(figsize = (9,5))

sns.lineplot(data=hourly_summary, x="hour", y="total_rentals", hue="day_type")
plt.title("Mean Hourly Bike Rentals by Day Type")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Total Rentals")

plt.tight_layout()
plt.savefig("output/hourly_demand.png", dpi=300)
plt.close()

# 4B - 4
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=clean_df,
    x="temperature_c",
    y="total_rentals",
    hue="season_label"
)

plt.title("Temperature vs. Hourly Bike Rentals by Season")
plt.xlabel("Temperature (°C)")
plt.ylabel("Total Rentals")

plt.tight_layout()
plt.savefig(output_folder / "temperature_scatterplot.png", dpi=300)
plt.close()


# Task 5
print("Task 5 A")
# 5A - 1
high_demand_threshold = clean_df["total_rentals"].quantile(0.75)
clean_df["high_demand"] = (clean_df["total_rentals"] >= high_demand_threshold)
print("\nHigh Demand Threshold:", round(high_demand_threshold), 2)

# 5A - 2
p_high = clean_df["high_demand"].mean()
print("\nP(High Demand):", round(p_high), 4)

# 5A - 3
working_rows = clean_df["workingday"] == 1

p_high_given_working = ((clean_df["high_demand"] & working_rows).sum() / working_rows.sum())

print("\nP(High Demand | Working Day):", round(p_high_given_working, 4))
print("Numerator:", (clean_df["high_demand"] & working_rows).sum())
print("Denominator:", working_rows.sum())

# 5A - 4
nonworking_rows = clean_df["workingday"] == 0

p_high_given_nonworking = (
    (clean_df["high_demand"] & nonworking_rows).sum()
    / nonworking_rows.sum()
)

print("\nP(High Demand | Nonworking Day):", round(p_high_given_nonworking, 4))
print("Numerator:", (clean_df["high_demand"] & nonworking_rows).sum())
print("Denominator:", nonworking_rows.sum())

#5A - 5
clear_rows = clean_df["weather_code"] == 1

p_high_given_clear = (
    (clean_df["high_demand"] & clear_rows).sum()
    / clear_rows.sum()
)

print("\nP(High Demand | Clear/Partly Cloudy):", round(p_high_given_clear, 4))
print("Numerator:", (clean_df["high_demand"] & clear_rows).sum())
print("Denominator:", clear_rows.sum())

print("TASK 5 B")
# 5B -1
correlation_columns = [
    "temperature_c",
    "apparent_temperature_c",
    "humidity_percent",
    "wind_speed",
    "total_rentals"
]

correlation_matrix = clean_df[correlation_columns].corr(method = "pearson")

print("\nPearson Correlation Matric:")
print(correlation_matrix.round(2))

plt.figure(figsize = (9, 6))

sns.heatmap(
    correlation_matrix,
    annot = True,
    fmt = ".2f",
    cmap = "coolwarm"
)

plt.title("Pearson Correlation Matrix of Weather and Bike Rentals")

plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png")
plt.close()

weather_correlations = correlation_matrix["total_rentals"].drop("total_rentals")

strongest_varibale = weather_correlations.abs().idxmax()
strongest_correlation = weather_correlations[strongest_varibale]

print("\nStrongest Weather Correlation:", strongest_varibale)
print("Correlation Coefficient:", round(strongest_correlation, 2))

# Task 6 
print("TASK 6 A")
# 6A
sample_df = clean_df.sample(n = 500, replace = False, random_state = 42)

full_mean = clean_df["total_rentals"].mean()
sample_mean = sample_df["total_rentals"].mean()

mean_difference = sample_mean - full_mean

print("Full Dataset mean:", round(full_mean, 2))
print("Sample mean:", round(sample_mean, 2))
print("Difference:", round(mean_difference, 2))

print("TASK 6 B")
# 6B
rng = np.random.default_rng(42)

sample_means = []
for i in range(200):
    sample = clean_df.sample(
        n= 500,
        replace = False,
        random_state = int(rng.integers(0, 1000000))
    )
    sample_means.append(sample["total_rentals"].mean())
    
mean_of_means = np.mean(sample_means)
std_of_means = np.std(sample_means, ddof = 1)

print("Mean of 200 Sample Means:", round(mean_of_means, 2))
print("Standard Deviation of Sample Means:", round(std_of_means, 2))

plt.figure(figsize = (8,5))

sns.histplot(sample_means, bins = 20)

plt.axvline(full_mean, color="red", linestyle = "--", label = "Full Dataset Mean")

plt.title("Distribution of 200 sample means")
plt.xlabel("Mean of Total Rentals")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(output_folder / "sampling_distribution.png", dpi = 300)
plt.close()

print("TASK 6 C")
# 6C
rentals = clean_df["total_rentals"].to_numpy()

bootstrap_means = []

for _ in range(5000):
    bootstrap_sample = rng.choice(rentals, size=len(rentals), replace=True)
    bootstrap_means.append(bootstrap_sample.mean())

bootstrap_se = np.std(bootstrap_means, ddof = 1)

lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])

print("Bootstrap Standard Error:", round(bootstrap_se, 2))
print("95% Confidence Interval:", round(lower, 2), "to", round(upper, 2))

print("TASK 6 D")
# 6D
clean_df.to_csv(output_folder / "bike sharing prepared.csv", index=False)

reloaded_df = pd.read_csv(output_folder / "bike sharing prepared.csv")


print("\nFirst 5 Reloaded Rows:")
print(reloaded_df.head())

print("\nReloaded Shape:", reloaded_df.shape)

print("\nMissing Values:")
print(reloaded_df.isna().sum())

assert reloaded_df.shape == clean_df.shape

print("\nExport verification successful!")
