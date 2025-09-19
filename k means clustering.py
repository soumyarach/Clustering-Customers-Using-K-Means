# ======================
# Step 1. Import Libraries
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================
# Step 2. Load Dataset
# ======================
# Example dataset: Customers with Age, Income, Spending Score
# (You can replace this with your own customer purchase data)
data = {
    "CustomerID": range(1, 11),
    "Age": [25, 34, 22, 45, 52, 36, 23, 44, 53, 48],
    "Annual_Income": [40000, 52000, 25000, 70000, 85000, 48000, 30000, 76000, 90000, 83000],
    "Spending_Score": [60, 65, 30, 80, 20, 70, 25, 75, 15, 18]
}
df = pd.DataFrame(data)

print("Sample Data:")
print(df.head())

# ======================
# Step 3. Normalize Features
# ======================
features = df[["Age", "Annual_Income", "Spending_Score"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ======================
# Step 4. Elbow Method to Choose k
# ======================
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal k")
plt.show()

# ======================
# Step 5. Fit KMeans with Optimal k (e.g., k=3 or 4 from elbow curve)
# ======================
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, init="k-means++", random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustered Data:")
print(df.head())

# ======================
# Step 6. Visualize Clusters
# ======================
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score",
                hue="Cluster", palette="Set1", s=100)
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:,1],  # Annual_Income
    scaler.inverse_transform(kmeans.cluster_centers_)[:,2],  # Spending_Score
    s=300, c="black", marker="X", label="Centroids"
)
plt.title("Customer Segmentation (Income vs Spending Score)")
plt.legend()
plt.show()

# ======================
# Step 7. Extra Visualization
# ======================
sns.pairplot(df, vars=["Age", "Annual_Income", "Spending_Score"], hue="Cluster", palette="Set1")
plt.suptitle("Cluster Distribution across Features", y=1.02)
plt.show()
