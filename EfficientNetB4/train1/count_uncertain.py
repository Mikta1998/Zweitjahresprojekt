import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load predictions
csv_path = "EfficientNetB4/train1/predictions.csv"
df = pd.read_csv(csv_path)

# Count total and retained per class
total_counts = df["True Label"].value_counts()
confident_df = df[df["Predicted Label"] != "Uncertain"]
confident_counts = confident_df["True Label"].value_counts()

# Combine results
coverage_df = pd.DataFrame({
    "Class": total_counts.index,
    "Total Samples": total_counts.values,
    "Confident Predictions": [confident_counts.get(cls, 0) for cls in total_counts.index]
})
coverage_df["Retained (%)"] = (coverage_df["Confident Predictions"] / coverage_df["Total Samples"] * 100).round(2)

# Save table
coverage_df.to_csv("EfficientNetB4/train1/class_retention_stats.csv", index=False)

# Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=coverage_df, x="Class", y="Retained (%)", palette="Blues_d")
plt.ylim(0, 100)
plt.title("Retention Rate per Class after Confidence Threshold (0.80)", fontsize=14, fontweight='bold')
plt.ylabel("Retained Predictions (%)")
plt.xlabel("Class")
plt.tight_layout()

# Save figure
plot_path = os.path.join("EfficientNetB4/train1", "class_retention_rate.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("Analysis complete.")
print("Saved table to: class_retention_stats.csv")
print("Saved plot to :", plot_path)