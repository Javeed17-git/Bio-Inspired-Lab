
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# --------------------------------------------------
# 1. Define the cities using (x, y) coordinates
# --------------------------------------------------
cities = np.array([
    [10, 20],
    [20, 30],
    [30, 15],
    [15, 10],
    [40, 25],
    [50, 10],
    [45, 35],
    [25, 45],
    [5, 35],
    [35, 40]
], dtype=float)

city_names = [f"City {i + 1}" for i in range(len(cities))]
N = len(cities)

# Calculate Euclidean distances between every pair of cities
distance = np.sqrt(
    ((cities[:, None, :] - cities[None, :, :]) ** 2).sum(axis=2)
)
np.fill_diagonal(distance, 1e-10)


# --------------------------------------------------
# 2. Calculate total length of a tour
# --------------------------------------------------
def tour_length(tour):
    total = 0.0

    for i in range(len(tour)):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total += distance[current_city, next_city]

    return total


# --------------------------------------------------
# 3. Construct one ant's tour probabilistically
# --------------------------------------------------
def construct_tour(pheromone, alpha, beta, rng):
    start = int(rng.integers(N))
    tour = [start]

    unvisited = set(range(N))
    unvisited.remove(start)

    while unvisited:
        current = tour[-1]
        candidates = sorted(unvisited)

        # Pheromone influence
        pheromone_values = pheromone[current, candidates] ** alpha

        # Heuristic influence: inverse distance
        heuristic_values = (
            1.0 / distance[current, candidates]
        ) ** beta

        # Calculate transition probabilities
        weights = pheromone_values * heuristic_values
        probabilities = weights / weights.sum()

        next_city = int(rng.choice(candidates, p=probabilities))

        tour.append(next_city)
        unvisited.remove(next_city)

    return tour


# --------------------------------------------------
# 4. Ant Colony Optimization algorithm
# --------------------------------------------------
def aco_tsp(
    num_ants=10,
    alpha=1.0,
    beta=3.0,
    rho=0.5,
    iterations=200,
    Q=100.0,
    seed=42
):
    rng = np.random.default_rng(seed)

    # Initialize pheromone trails
    pheromone = np.ones((N, N), dtype=float)

    best_tour = None
    best_distance = float("inf")
    convergence = []

    start_time = perf_counter()

    for iteration in range(iterations):
        all_tours = []
        all_lengths = []

        # Each ant constructs a tour
        for ant in range(num_ants):
            tour = construct_tour(
                pheromone, alpha, beta, rng
            )

            length = tour_length(tour)

            all_tours.append(tour)
            all_lengths.append(length)

            # Store the best tour
            if length < best_distance:
                best_distance = length
                best_tour = tour.copy()

        # Record the best distance at this iteration
        convergence.append(best_distance)

        # Evaporate pheromone
        pheromone *= (1.0 - rho)

        # Deposit pheromone based on tour quality
        for tour, length in zip(all_tours, all_lengths):
            deposit = Q / length

            for i in range(N):
                a = tour[i]
                b = tour[(i + 1) % N]

                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

    elapsed_time = perf_counter() - start_time

    return best_tour, best_distance, elapsed_time, convergence


# --------------------------------------------------
# 5. Compare three ACO configurations
# --------------------------------------------------
configurations = [
    {
        "name": "ACO - Baseline",
        "ants": 10,
        "alpha": 1.0,
        "beta": 3.0,
        "rho": 0.5,
        "iterations": 200
    },
    {
        "name": "ACO - More Ants",
        "ants": 20,
        "alpha": 1.0,
        "beta": 3.0,
        "rho": 0.5,
        "iterations": 200
    },
    {
        "name": "ACO - More Exploration",
        "ants": 10,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.3,
        "iterations": 200
    }
]

results = []
best_overall = None

for index, config in enumerate(configurations):
    tour, length, runtime, history = aco_tsp(
        num_ants=config["ants"],
        alpha=config["alpha"],
        beta=config["beta"],
        rho=config["rho"],
        iterations=config["iterations"],
        seed=42 + index
    )

    result = {
        "name": config["name"],
        "distance": length,
        "runtime": runtime,
        "tour": tour,
        "history": history
    }

    results.append(result)

    if best_overall is None or length < best_overall["distance"]:
        best_overall = result


# --------------------------------------------------
# 6. Print performance comparison
# --------------------------------------------------
print("\nACO PERFORMANCE COMPARISON")
print("-" * 65)
print(f"{'Configuration':<27} {'Distance':>13} {'Time (s)':>13}")
print("-" * 65)

for result in results:
    print(
        f"{result['name']:<27}"
        f"{result['distance']:>13.3f}"
        f"{result['runtime']:>13.6f}"
    )

print("-" * 65)

# Display the best route
route = best_overall["tour"]
closed_route = route + [route[0]]

print("\nBEST ROUTE FOUND")
print(" -> ".join(city_names[i] for i in closed_route))
print(f"Total distance: {best_overall['distance']:.3f}")
print(f"Execution time: {best_overall['runtime']:.6f} seconds")


# --------------------------------------------------
# 7. Plot convergence graph
# --------------------------------------------------
plt.figure(figsize=(9, 5))

for result in results:
    plt.plot(
        range(1, len(result["history"]) + 1),
        result["history"],
        label=result["name"]
    )

plt.xlabel("Iteration")
plt.ylabel("Best Tour Distance")
plt.title("ACO Convergence for the Travelling Salesman Problem")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Plot the best route with directional arrows
# --------------------------------------------------
plt.figure(figsize=(9, 7))

# Draw the complete tour
x = cities[closed_route, 0]
y = cities[closed_route, 1]

plt.plot(
    x, y,
    color="royalblue",
    linewidth=1.8,
    label="Travel route"
)

# Add arrows to indicate travel direction
for i in range(len(closed_route) - 1):
    x1, y1 = cities[closed_route[i]]
    x2, y2 = cities[closed_route[i + 1]]

    # Arrow positioned along the middle of each route segment
    plt.annotate(
        "",
        xy=(
            x1 + 0.65 * (x2 - x1),
            y1 + 0.65 * (y2 - y1)
        ),
        xytext=(
            x1 + 0.35 * (x2 - x1),
            y1 + 0.35 * (y2 - y1)
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=1.8,
            mutation_scale=15
        )
    )

# Plot all cities
plt.scatter(
    cities[:, 0],
    cities[:, 1],
    color="darkgreen",
    s=75,
    zorder=3,
    label="Cities"
)

# Label each city
for i, (cx, cy) in enumerate(cities):
    plt.annotate(
        city_names[i],
        (cx, cy),
        xytext=(5, 7),
        textcoords="offset points"
    )

# Highlight the starting city
start = closed_route[0]

plt.scatter(
    cities[start, 0],
    cities[start, 1],
    color="orange",
    edgecolors="black",
    s=160,
    zorder=4,
    label="Starting City"
)

plt.title(
    f"Best ACO Tour | Total Distance = "
    f"{best_overall['distance']:.3f}"
)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.tight_layout()
plt.show()