import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load scores
    df = pd.read_csv("artifacts/scores.csv")

    print("\n=== Dataset Summary ===")
    print(df.describe())

    # Show top anomalies
    print("\n=== Top 10 Anomalous Sessions ===")
    print(df.sort_values("hybrid_score", ascending=False).head(10))

    # Plot score distribution
    plt.hist(df["hybrid_score"], bins=30, edgecolor="black")
    plt.title("Hybrid Anomaly Score Distribution")
    plt.xlabel("Hybrid Score")
    plt.ylabel("Count")
    plt.savefig("artifacts/reports/score_distribution.png")
    plt.show()

    # Save ranked anomalies
    df_sorted = df.sort_values("hybrid_score", ascending=False)
    df_sorted.to_csv("artifacts/reports/ranked_anomalies.csv", index=False)
    print("\nSaved ranked anomalies to artifacts/reports/ranked_anomalies.csv")
    print("Saved score distribution plot to artifacts/reports/score_distribution.png")

if __name__ == "__main__":
    main()
