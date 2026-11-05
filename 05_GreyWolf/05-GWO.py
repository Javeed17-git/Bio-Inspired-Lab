
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. CREATE INDUSTRIAL SENSOR DATASET
# ==========================================

np.random.seed(42)

# Normal sensor data
# Temperature, Pressure, Vibration, Speed, Voltage

normal_data = np.column_stack([
    np.random.uniform(25, 45, 1000),
    np.random.uniform(20, 40, 1000),
    np.random.uniform(1, 4, 1000),
    np.random.uniform(700, 1300, 1000),
    np.random.uniform(210, 235, 1000)
])

# Abnormal sensor data

anomaly_data = np.column_stack([
    np.random.uniform(55, 90, 200),
    np.random.uniform(55, 75, 200),
    np.random.uniform(6, 12, 200),
    np.random.uniform(1600, 2000, 200),
    np.random.uniform(250, 280, 200)
])

# ==========================================
# 2. HARD-CODED SENSOR THRESHOLDS
# ==========================================

LOWER_LIMITS = np.array([
    20,     # Temperature °C
    10,     # Pressure PSI
    0,      # Vibration mm/s
    500,    # Motor speed RPM
    200     # Voltage V
])

UPPER_LIMITS = np.array([
    50,     # Temperature °C
    50,     # Pressure PSI
    5,      # Vibration mm/s
    1500,   # Motor speed RPM
    240     # Voltage V
])

SENSOR_NAMES = [
    "Temperature",
    "Pressure",
    "Vibration",
    "Motor Speed",
    "Voltage"
]

print("Industrial Sensor Thresholds")

for i in range(5):
    print(
        SENSOR_NAMES[i],
        ":",
        LOWER_LIMITS[i],
        "to",
        UPPER_LIMITS[i]
    )


# ==========================================
# 3. THRESHOLD DETECTION FUNCTION
# ==========================================

def threshold_detection(data):

    anomaly = np.any(
        (data < LOWER_LIMITS) |
        (data > UPPER_LIMITS),
        axis=1
    )

    return anomaly.astype(int)


# ==========================================
# 4. SPLIT DATASET
# ==========================================

train_normal = normal_data[:800]

validation_normal = normal_data[800:900]
validation_anomaly = anomaly_data[:100]

test_normal = normal_data[900:1000]
test_anomaly = anomaly_data[100:200]

validation_raw = np.vstack([
    validation_normal,
    validation_anomaly
])

validation_y = np.concatenate([
    np.zeros(100),
    np.ones(100)
]).astype(int)

test_raw = np.vstack([
    test_normal,
    test_anomaly
])

test_y = np.concatenate([
    np.zeros(100),
    np.ones(100)
]).astype(int)

# ==========================================
# 5. DATA NORMALIZATION
# ==========================================

scaler = StandardScaler()

train_normal_scaled = scaler.fit_transform(
    train_normal
)

validation_X = scaler.transform(
    validation_raw
)

test_X = scaler.transform(
    test_raw
)

print("\nTraining samples:", len(train_normal))
print("Validation samples:", len(validation_X))
print("Testing samples:", len(test_X))


# ==========================================
# 6. FITNESS FUNCTION
# ==========================================

def fitness_function(position):

    contamination = np.clip(
        position[0], 0.01, 0.30
    )

    n_estimators = int(
        np.clip(position[1], 50, 250)
    )

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(train_normal_scaled)

    predictions = model.predict(validation_X)

    # Convert Isolation Forest output
    # -1 = Anomaly, 1 = Normal

    predictions = np.where(
        predictions == -1, 1, 0
    )

    # Apply fixed sensor thresholds

    threshold_predictions = threshold_detection(
        validation_raw
    )

    # Combine both methods

    final_predictions = np.maximum(
        predictions,
        threshold_predictions
    )

    f1 = f1_score(
        validation_y,
        final_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        validation_y,
        final_predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Fitness function
    # Objective 1: Maximize F1-score
    # Objective 2: Minimize False Positive Rate

    fitness = 0.7 * (1 - f1) + 0.3 * fpr

    return fitness


# ==========================================
# 7. GREY WOLF OPTIMIZER
# ==========================================

def GWO(population_size=15, iterations=30):

    dimensions = 2

    lower_bound = np.array([0.01, 50])
    upper_bound = np.array([0.30, 250])

    population = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    alpha = np.zeros(dimensions)
    beta = np.zeros(dimensions)
    delta = np.zeros(dimensions)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    convergence = []

    for t in range(iterations):

        for i in range(population_size):

            score = fitness_function(population[i])

            if score < alpha_score:

                delta_score = beta_score
                delta = beta.copy()

                beta_score = alpha_score
                beta = alpha.copy()

                alpha_score = score
                alpha = population[i].copy()

            elif score < beta_score:

                delta_score = beta_score
                delta = beta.copy()

                beta_score = score
                beta = population[i].copy()

            elif score < delta_score:

                delta_score = score
                delta = population[i].copy()

        # Decrease exploration parameter

        a = 2 - 2 * t / iterations

        new_population = np.zeros_like(population)

        for i in range(population_size):

            positions = []

            for leader in [alpha, beta, delta]:

                r1 = np.random.rand(dimensions)
                r2 = np.random.rand(dimensions)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(
                    C * leader - population[i]
                )

                new_position = leader - A * D

                positions.append(new_position)

            new_population[i] = (
                positions[0]
                + positions[1]
                + positions[2]
            ) / 3

        population = np.clip(
            new_population,
            lower_bound,
            upper_bound
        )

        convergence.append(alpha_score)

        print(
            f"Iteration {t+1}/{iterations} "
            f"| Best Fitness: {alpha_score:.5f}"
        )

    return alpha, convergence


# ==========================================
# 8. RUN GWO
# ==========================================

best_parameters, convergence = GWO(
    population_size=15,
    iterations=30
)

best_contamination = np.clip(
    best_parameters[0], 0.01, 0.30
)

best_estimators = int(
    np.clip(best_parameters[1], 50, 250)
)

print("\nOptimized Parameters")

print("Contamination:", best_contamination)
print("Number of Trees:", best_estimators)


# ==========================================
# 9. BASELINE ISOLATION FOREST
# ==========================================

baseline_model = IsolationForest(
    contamination=0.10,
    n_estimators=100,
    random_state=42
)

baseline_model.fit(train_normal_scaled)

baseline_if = baseline_model.predict(test_X)

baseline_if = np.where(
    baseline_if == -1, 1, 0
)

baseline_threshold = threshold_detection(
    test_raw
)

baseline_predictions = np.maximum(
    baseline_if,
    baseline_threshold
)


# ==========================================
# 10. GWO OPTIMIZED ISOLATION FOREST
# ==========================================

gwo_model = IsolationForest(
    contamination=best_contamination,
    n_estimators=best_estimators,
    random_state=42
)

gwo_model.fit(train_normal_scaled)

gwo_if = gwo_model.predict(test_X)

gwo_if = np.where(
    gwo_if == -1, 1, 0
)

gwo_threshold = threshold_detection(
    test_raw
)

gwo_predictions = np.maximum(
    gwo_if,
    gwo_threshold
)


# ==========================================
# 11. EVALUATION FUNCTION
# ==========================================

def evaluate_model(y_true, predictions):

    accuracy = accuracy_score(
        y_true, predictions
    )

    precision = precision_score(
        y_true, predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true, predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true, predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100,
        fpr * 100
    ]


# ==========================================
# 12. COMPARE RESULTS
# ==========================================

baseline_results = evaluate_model(
    test_y, baseline_predictions
)

gwo_results = evaluate_model(
    test_y, gwo_predictions
)

results = pd.DataFrame({

    "Metrics": [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "False Positive Rate (%)"
    ],

    "Isolation Forest + Threshold":
        baseline_results,

    "GWO + Isolation Forest + Threshold":
        gwo_results

})

print("\n========== FINAL RESULTS ==========")

print(results.to_string(index=False))


# ==========================================
# 13. CONVERGENCE GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(convergence) + 1),
    convergence,
    color="blue",
    marker="o"
)

plt.title("GWO Convergence Curve")
plt.xlabel("Iterations")
plt.ylabel("Fitness Value")
plt.grid(True)

plt.show()


# ==========================================
# 14. ACCURACY COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["IF + Threshold", "GWO + IF + Threshold"],
    [baseline_results[0], gwo_results[0]],
    color=["orange", "green"]
)

plt.ylabel("Accuracy (%)")
plt.title("Anomaly Detection Accuracy Comparison")

plt.show()


# ==========================================
# 15. FALSE POSITIVE COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["IF + Threshold", "GWO + IF + Threshold"],
    [baseline_results[4], gwo_results[4]],
    color=["orange", "green"]
)

plt.ylabel("False Positive Rate (%)")
plt.title("False Positive Rate Comparison")

plt.show()


# ==========================================
# 16. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    test_y,
    gwo_predictions
)

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title("GWO Anomaly Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Normal", "Anomaly"])
plt.yticks([0, 1], ["Normal", "Anomaly"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.colorbar()
plt.show()


# ==========================================
# 17. TEST NEW INDUSTRIAL SENSOR READINGS
# ==========================================

new_sensor_readings = np.array([

    [40, 30, 3, 1000, 220],

    [60, 30, 3, 1000, 220],

    [35, 55, 3, 1000, 220],

    [40, 30, 8, 1000, 220],

    [40, 30, 3, 1800, 220],

    [40, 30, 3, 1000, 260]

])

new_scaled = scaler.transform(
    new_sensor_readings
)

if_predictions = gwo_model.predict(new_scaled)

if_predictions = np.where(
    if_predictions == -1, 1, 0
)

threshold_predictions = threshold_detection(
    new_sensor_readings
)

final_predictions = np.maximum(
    if_predictions,
    threshold_predictions
)

print("\n========== NEW SENSOR DETECTION ==========")

for i, reading in enumerate(new_sensor_readings):

    print("\nSensor Reading:", reading)

    if final_predictions[i] == 1:

        print("STATUS: ANOMALY DETECTED")

        for j, value in enumerate(reading):

            if value < LOWER_LIMITS[j]:

                print(
                    SENSOR_NAMES[j],
                    "is BELOW the lower limit"
                )

            elif value > UPPER_LIMITS[j]:

                print(
                    SENSOR_NAMES[j],
                    "is ABOVE the upper limit"
                )

    else:

        print("STATUS: NORMAL")