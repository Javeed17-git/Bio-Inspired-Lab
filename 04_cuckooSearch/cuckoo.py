
import time
import math
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin


# ==========================================================
# LOAD AND PREPROCESS DATA
# ==========================================================
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

K = 3                 # Number of clusters
N_NESTS = 20          # Number of candidate solutions
MAX_ITER = 100        # Maximum iterations
PA = 0.25             # Probability of abandoning nests
ALPHA = 0.01          # Lévy flight step-size factor

rng = np.random.default_rng(42)
LOW = X.min(axis=0)
HIGH = X.max(axis=0)
N_FEATURES = X.shape[1]


# ==========================================================
# FITNESS FUNCTION: WITHIN-CLUSTER SUM OF SQUARES (WCSS)
# ==========================================================
def fitness(centers):
    labels = pairwise_distances_argmin(X, centers)
    distances = X - centers[labels]
    return np.sum(distances ** 2)


# ==========================================================
# LEVY FLIGHT USING MANTEGNA'S ALGORITHM
# ==========================================================
def levy_flight(shape, beta=1.5):
    sigma = (
        math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = rng.normal(0, sigma, shape)
    v = rng.normal(0, 1, shape)

    return u / (np.abs(v) ** (1 / beta) + 1e-12)


# ==========================================================
# CUCKOO SEARCH CLUSTERING
# ==========================================================
def cuckoo_search():
    # Each nest contains K cluster centers.
    nests = np.array([
        X[rng.choice(len(X), K, replace=False)].copy()
        for _ in range(N_NESTS)
    ])

    scores = np.array([fitness(nest) for nest in nests])

    best_idx = np.argmin(scores)
    best_nest = nests[best_idx].copy()
    best_score = scores[best_idx]

    start = time.perf_counter()

    for iteration in range(MAX_ITER):

        # Generate new candidate solutions via Lévy flights.
        for i in range(N_NESTS):
            step = levy_flight(nests[i].shape)

            candidate = (
                nests[i]
                + ALPHA * step * (nests[i] - best_nest)
            )

            candidate = np.clip(candidate, LOW, HIGH)
            candidate_score = fitness(candidate)

            # Replace a randomly chosen nest if improved.
            j = rng.integers(N_NESTS)

            if candidate_score < scores[j]:
                nests[j] = candidate
                scores[j] = candidate_score

        # Abandon a fraction of the worst nests.
        n_abandon = max(1, int(PA * N_NESTS))
        worst = np.argsort(scores)[-n_abandon:]

        for j in worst:
            nests[j] = X[
                rng.choice(len(X), K, replace=False)
            ].copy()
            scores[j] = fitness(nests[j])

        # Preserve the best solution found so far.
        current_idx = np.argmin(scores)

        if scores[current_idx] < best_score:
            best_score = scores[current_idx]
            best_nest = nests[current_idx].copy()

        if (iteration + 1) % 20 == 0:
            print(
                f"Iteration {iteration + 1}: "
                f"Best WCSS = {best_score:.4f}"
            )

    elapsed = time.perf_counter() - start

    return best_nest, elapsed


# ==========================================================
# OBJECTIVE 1: OPTIMIZE CLUSTER CENTERS
# ==========================================================
print("\n--- Objective 1: Cuckoo Search Clustering ---")

cs_centers, cs_time = cuckoo_search()

cs_labels = pairwise_distances_argmin(X, cs_centers)
cs_wcss = fitness(cs_centers)
cs_silhouette = silhouette_score(X, cs_labels)

print("\nCuckoo Search cluster centers:")
print(cs_centers)

print(f"WCSS: {cs_wcss:.4f}")
print(f"Silhouette Score: {cs_silhouette:.4f}")


# ==========================================================
# OBJECTIVE 2: COMPARE AGAINST K-MEANS
# ==========================================================
print("\n--- Objective 2: Comparison with K-means ---")

start = time.perf_counter()

kmeans = KMeans(
    n_clusters=K,
    n_init=10,
    random_state=42
)
km_labels = kmeans.fit_predict(X)

km_time = time.perf_counter() - start
km_wcss = kmeans.inertia_
km_silhouette = silhouette_score(X, km_labels)

results = pd.DataFrame({
    "Metric": [
        "WCSS (lower is better)",
        "Silhouette (higher is better)",
        "Execution time (seconds)"
    ],
    "Cuckoo Search": [
        cs_wcss, cs_silhouette, cs_time
    ],
    "K-means": [
        km_wcss, km_silhouette, km_time
    ]
})

print("\nComparison results:")
print(results.to_string(index=False))

print("\nInterpretation:")
if cs_wcss < km_wcss:
    print("Cuckoo Search achieved lower WCSS.")
else:
    print("K-means achieved equal or lower WCSS.")

if cs_silhouette > km_silhouette:
    print("Cuckoo Search achieved a higher Silhouette Score.")
else:
    print("K-means achieved an equal or higher Silhouette Score.")


   