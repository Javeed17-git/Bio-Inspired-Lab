import numpy as np
import random

# ============================================================
# GENETIC ALGORITHM FOR FINANCIAL PORTFOLIO OPTIMIZATION
#
# Objectives:
# 1. Maximize expected portfolio return
# 2. Minimize portfolio risk (variance)
#
# Based on the risk-return portfolio optimization approach
# discussed in IEEE research on GA-based portfolio optimization.
# ============================================================


# ------------------------------------------------------------
# 1. SAMPLE HISTORICAL RETURNS
# ------------------------------------------------------------
# Each row represents one trading period.
# Each column represents one asset.
#
# In a real project, these values can be replaced with
# historical stock returns.
# ------------------------------------------------------------

np.random.seed(42)

stocks = [
    "Stock A",
    "Stock B",
    "Stock C",
    "Stock D",
    "Stock E"
]

# Simulated daily returns
returns = np.random.normal(
    loc=[0.0008, 0.0006, 0.0010, 0.0005, 0.0009],
    scale=[0.015, 0.012, 0.020, 0.010, 0.018],
    size=(252, 5)
)

# ------------------------------------------------------------
# 2. CALCULATE EXPECTED RETURN AND COVARIANCE
# ------------------------------------------------------------

mean_returns = np.mean(returns, axis=0)
cov_matrix = np.cov(returns, rowvar=False)


# ------------------------------------------------------------
# 3. PORTFOLIO FUNCTIONS
# ------------------------------------------------------------

def portfolio_return(weights):
    """
    Calculate expected portfolio return.
    """

    return np.sum(weights * mean_returns)


def portfolio_risk(weights):
    """
    Calculate portfolio variance.
    """

    return np.dot(
        weights.T,
        np.dot(cov_matrix, weights)
    )


def normalize_weights(weights):
    """
    Make sure portfolio weights add up to 1.
    """

    weights = np.maximum(weights, 0)

    total = np.sum(weights)

    if total == 0:
        return np.ones(len(weights)) / len(weights)

    return weights / total


# ------------------------------------------------------------
# 4. FITNESS FUNCTION
# ------------------------------------------------------------

def fitness(weights):
    """
    Fitness function combines return and risk.

    Higher return is good.
    Lower risk is good.

    Therefore:

        Fitness = Return - Risk

    A risk-aversion parameter is used to control the
    importance of risk.
    """

    risk_aversion = 5

    ret = portfolio_return(weights)
    risk = portfolio_risk(weights)

    return ret - risk_aversion * risk


# ------------------------------------------------------------
# 5. CREATE INITIAL POPULATION
# ------------------------------------------------------------

def create_population(population_size, number_of_assets):

    population = []

    for _ in range(population_size):

        weights = np.random.random(number_of_assets)

        weights = normalize_weights(weights)

        population.append(weights)

    return population


# ------------------------------------------------------------
# 6. SELECTION
# ------------------------------------------------------------

def selection(population):
    """
    Tournament selection.

    Randomly select two individuals and keep the one
    with better fitness.
    """

    candidate1 = random.choice(population)
    candidate2 = random.choice(population)

    if fitness(candidate1) > fitness(candidate2):
        return candidate1.copy()
    else:
        return candidate2.copy()


# ------------------------------------------------------------
# 7. CROSSOVER
# ------------------------------------------------------------

def crossover(parent1, parent2):
    """
    Arithmetic crossover.

    A new child is created using a combination of
    two parents.
    """

    alpha = random.random()

    child = alpha * parent1 + (1 - alpha) * parent2

    return normalize_weights(child)


# ------------------------------------------------------------
# 8. MUTATION
# ------------------------------------------------------------

def mutation(child, mutation_rate=0.1):
    """
    Randomly change portfolio weights.
    """

    for i in range(len(child)):

        if random.random() < mutation_rate:

            child[i] += np.random.normal(0, 0.05)

    return normalize_weights(child)


# ------------------------------------------------------------
# 9. GENETIC ALGORITHM
# ------------------------------------------------------------

def genetic_algorithm(
    population_size=100,
    generations=200,
    mutation_rate=0.1
):

    number_of_assets = len(stocks)

    # Create initial population
    population = create_population(
        population_size,
        number_of_assets
    )

    best_solution = None
    best_fitness = -float("inf")

    # --------------------------------------------------------
    # EVOLUTION
    # --------------------------------------------------------

    for generation in range(generations):

        new_population = []

        # Find best individual in current population
        current_best = max(
            population,
            key=fitness
        )

        current_fitness = fitness(current_best)

        if current_fitness > best_fitness:

            best_fitness = current_fitness

            best_solution = current_best.copy()

        # Create next generation
        while len(new_population) < population_size:

            parent1 = selection(population)
            parent2 = selection(population)

            child = crossover(
                parent1,
                parent2
            )

            child = mutation(
                child,
                mutation_rate
            )

            new_population.append(child)

        population = new_population

    return best_solution


# ------------------------------------------------------------
# 10. RUN GENETIC ALGORITHM
# ------------------------------------------------------------

best_weights = genetic_algorithm()


# ------------------------------------------------------------
# 11. DISPLAY RESULTS
# ------------------------------------------------------------

expected_return = portfolio_return(best_weights)

risk = portfolio_risk(best_weights)

print("\n======================================")
print("OPTIMAL PORTFOLIO")
print("======================================")

for stock, weight in zip(stocks, best_weights):

    print(
        f"{stock}: {weight * 100:.2f}%"
    )

print("--------------------------------------")

print(
    f"Expected Daily Return: "
    f"{expected_return:.6f}"
)

print(
    f"Portfolio Risk (Variance): "
    f"{risk:.6f}"
)

print(
    f"Fitness Value: "
    f"{fitness(best_weights):.6f}"
)

print("======================================")