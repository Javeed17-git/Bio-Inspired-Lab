import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# PARTICLE SWARM OPTIMIZATION FOR ROBOT PATH PLANNING
#
# Based on:
# Masehian & Sedighizadeh (2010)
# "Multi-objective robot motion planning using a
# particle swarm optimization model"
#
# Objectives:
#   1. Minimize path length
#   2. Minimize path oscillation / maximize smoothness
#
# ============================================================


# ------------------------------------------------------------
# 1. ENVIRONMENT
# ------------------------------------------------------------

START = np.array([1.0, 1.0])
GOAL  = np.array([9.0, 9.0])

# Rectangular obstacles:
# (x_min, y_min, width, height)
OBSTACLES = [
    (3.0, 1.5, 1.5, 4.0),
    (5.5, 5.0, 1.5, 3.0),
    (3.5, 6.5, 1.5, 1.5),
]

X_LIMITS = (0, 10)
Y_LIMITS = (0, 10)


# ------------------------------------------------------------
# 2. PSO PARAMETERS
# ------------------------------------------------------------

N_PARTICLES = 60
N_WAYPOINTS = 8
MAX_ITER = 250

# PSO parameters
W_MAX = 0.9
W_MIN = 0.4

C1 = 1.7       # cognitive coefficient
C2 = 1.7       # social coefficient

V_MAX = 0.7

# Objective weights
LAMBDA_LENGTH = 1.0
LAMBDA_SMOOTH = 0.25

# Collision penalty
COLLISION_PENALTY = 1000.0

# Reproducibility
np.random.seed(42)


# ------------------------------------------------------------
# 3. PARTICLE REPRESENTATION
# ------------------------------------------------------------

# Each particle contains N_WAYPOINTS intermediate points.
#
# Example:
#
# START --> P1 --> P2 --> P3 --> ... --> P8 --> GOAL
#
# Particle dimension:
#     N_WAYPOINTS * 2
#
# because every waypoint has (x,y).


DIMENSION = N_WAYPOINTS * 2


def particle_to_path(particle):
    """
    Convert a flattened PSO particle into a complete path.
    """

    waypoints = particle.reshape(N_WAYPOINTS, 2)

    path = np.vstack([
        START,
        waypoints,
        GOAL
    ])

    return path


# ------------------------------------------------------------
# 4. COLLISION DETECTION
# ------------------------------------------------------------

def point_inside_rectangle(point, rect):
    """
    Check whether a point lies inside a rectangular obstacle.
    """

    x, y = point
    xmin, ymin, width, height = rect

    xmax = xmin + width
    ymax = ymin + height

    return (
        xmin <= x <= xmax and
        ymin <= y <= ymax
    )


def orientation(a, b, c):
    """
    Orientation of three points.
    """

    value = (
        (b[1] - a[1]) * (c[0] - b[0])
        -
        (b[0] - a[0]) * (c[1] - b[1])
    )

    if abs(value) < 1e-10:
        return 0

    return 1 if value > 0 else 2


def segments_intersect(p1, q1, p2, q2):
    """
    Check if two line segments intersect.
    """

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    return False


def segment_intersects_rectangle(p1, p2, rect):
    """
    Check whether a line segment intersects
    a rectangular obstacle.
    """

    xmin, ymin, width, height = rect

    xmax = xmin + width
    ymax = ymin + height

    # Rectangle corners
    corners = [
        np.array([xmin, ymin]),
        np.array([xmax, ymin]),
        np.array([xmax, ymax]),
        np.array([xmin, ymax])
    ]

    # If either endpoint is inside
    if point_inside_rectangle(p1, rect):
        return True

    if point_inside_rectangle(p2, rect):
        return True

    # Rectangle edges
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0])
    ]

    for edge_start, edge_end in edges:

        if segments_intersect(
            p1, p2,
            edge_start, edge_end
        ):
            return True

    return False


def path_has_collision(path):
    """
    Check the complete path against all obstacles.
    """

    for i in range(len(path) - 1):

        p1 = path[i]
        p2 = path[i + 1]

        for obstacle in OBSTACLES:

            if segment_intersects_rectangle(
                p1, p2, obstacle
            ):
                return True

    return False


# ------------------------------------------------------------
# 5. OBJECTIVE 1: PATH LENGTH
# ------------------------------------------------------------

def path_length(path):
    """
    Calculate total Euclidean path length.
    """

    distances = np.linalg.norm(
        np.diff(path, axis=0),
        axis=1
    )

    return np.sum(distances)


# ------------------------------------------------------------
# 6. OBJECTIVE 2: PATH SMOOTHNESS
# ------------------------------------------------------------

def path_smoothness(path):
    """
    Calculate total turning angle.

    Smaller value = smoother path.
    """

    total_angle = 0.0

    for i in range(1, len(path) - 1):

        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            continue

        cosine = np.dot(v1, v2) / (norm1 * norm2)

        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.arccos(cosine)

        total_angle += abs(angle)

    return total_angle


# ------------------------------------------------------------
# 7. FITNESS FUNCTION
# ------------------------------------------------------------

def fitness(particle):
    """
    Combined objective function:

        F = lambda1 * normalized_length
            +
            lambda2 * normalized_smoothness

    Collision paths receive a large penalty.
    """

    path = particle_to_path(particle)

    # Objective 1
    length = path_length(path)

    # Objective 2
    smoothness = path_smoothness(path)

    # Normalize approximately to comparable scales
    normalized_length = length / 20.0
    normalized_smoothness = smoothness / np.pi

    cost = (
        LAMBDA_LENGTH * normalized_length
        +
        LAMBDA_SMOOTH * normalized_smoothness
    )

    # Collision constraint
    if path_has_collision(path):
        cost += COLLISION_PENALTY

    return cost


# ------------------------------------------------------------
# 8. INITIALIZE SWARM
# ------------------------------------------------------------

particles = np.random.uniform(
    low=[X_LIMITS[0], Y_LIMITS[0]],
    high=[X_LIMITS[1], Y_LIMITS[1]],
    size=(N_PARTICLES, N_WAYPOINTS, 2)
)

particles = particles.reshape(
    N_PARTICLES,
    DIMENSION
)

velocities = np.random.uniform(
    -V_MAX,
    V_MAX,
    size=(N_PARTICLES, DIMENSION)
)


# ------------------------------------------------------------
# 9. PERSONAL BEST
# ------------------------------------------------------------

pbest_positions = particles.copy()

pbest_scores = np.array([
    fitness(p)
    for p in particles
])


# ------------------------------------------------------------
# 10. GLOBAL BEST
# ------------------------------------------------------------

best_index = np.argmin(pbest_scores)

gbest_position = pbest_positions[
    best_index
].copy()

gbest_score = pbest_scores[
    best_index
]


# ------------------------------------------------------------
# 11. PSO ITERATION
# ------------------------------------------------------------

history = []

length_history = []
smooth_history = []


for iteration in range(MAX_ITER):

    # Linearly decreasing inertia weight
    w = (
        W_MAX
        -
        (W_MAX - W_MIN)
        *
        iteration / MAX_ITER
    )

    for i in range(N_PARTICLES):

        r1 = np.random.random(DIMENSION)
        r2 = np.random.random(DIMENSION)

        # Standard PSO velocity equation
        velocities[i] = (
            w * velocities[i]
            +
            C1 * r1 *
            (pbest_positions[i] - particles[i])
            +
            C2 * r2 *
            (gbest_position - particles[i])
        )

        # Velocity constraint
        velocities[i] = np.clip(
            velocities[i],
            -V_MAX,
            V_MAX
        )

        # Position update
        particles[i] += velocities[i]

        # Position constraint
        particles[i][0::2] = np.clip(
            particles[i][0::2],
            X_LIMITS[0],
            X_LIMITS[1]
        )

        particles[i][1::2] = np.clip(
            particles[i][1::2],
            Y_LIMITS[0],
            Y_LIMITS[1]
        )

        # Evaluate
        current_score = fitness(
            particles[i]
        )

        # Update personal best
        if current_score < pbest_scores[i]:

            pbest_scores[i] = current_score

            pbest_positions[i] = particles[i].copy()

        # Update global best
        if current_score < gbest_score:

            gbest_score = current_score

            gbest_position = particles[i].copy()

    # Store convergence
    best_path = particle_to_path(
        gbest_position
    )

    current_length = path_length(best_path)
    current_smoothness = path_smoothness(best_path)

    history.append(gbest_score)
    length_history.append(current_length)
    smooth_history.append(current_smoothness)

    if iteration % 25 == 0:

        print(
            f"Iteration {iteration:3d} | "
            f"Fitness = {gbest_score:.4f} | "
            f"Length = {current_length:.4f} | "
            f"Smoothness = {current_smoothness:.4f}"
        )


# ------------------------------------------------------------
# 12. FINAL RESULT
# ------------------------------------------------------------

best_path = particle_to_path(
    gbest_position
)

final_length = path_length(best_path)

final_smoothness = path_smoothness(
    best_path
)

collision = path_has_collision(
    best_path
)


print("\n==============================")
print("       FINAL RESULT")
print("==============================")

print(
    f"Path length       : {final_length:.4f}"
)

print(
    f"Smoothness/angle  : {final_smoothness:.4f} rad"
)

print(
    f"Collision free    : {not collision}"
)

print(
    f"Fitness            : {gbest_score:.4f}"
)


# ------------------------------------------------------------
# 13. PLOT OPTIMIZED PATH
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1, 2,
    figsize=(14, 6)
)


# ----- PATH PLOT -----

ax[0].set_title(
    "PSO Multi-Objective Robot Path"
)

# Draw obstacles
for obstacle in OBSTACLES:

    xmin, ymin, width, height = obstacle

    rectangle = Rectangle(
        (xmin, ymin),
        width,
        height,
        color="red",
        alpha=0.5
    )

    ax[0].add_patch(rectangle)


# Plot optimized path
ax[0].plot(
    best_path[:, 0],
    best_path[:, 1],
    "b-o",
    linewidth=2,
    markersize=4,
    label="PSO path"
)


# Start
ax[0].scatter(
    START[0],
    START[1],
    color="green",
    s=120,
    marker="o",
    label="Start"
)


# Goal
ax[0].scatter(
    GOAL[0],
    GOAL[1],
    color="black",
    s=120,
    marker="*",
    label="Goal"
)


ax[0].set_xlim(
    X_LIMITS
)

ax[0].set_ylim(
    Y_LIMITS
)

ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")

ax[0].grid(True)
ax[0].legend()


# ------------------------------------------------------------
# 14. CONVERGENCE GRAPH
# ------------------------------------------------------------

ax[1].plot(
    history,
    color="purple",
    linewidth=2
)

ax[1].set_title(
    "PSO Fitness Convergence"
)

ax[1].set_xlabel(
    "Iteration"
)

ax[1].set_ylabel(
    "Fitness"
)

ax[1].grid(True)


plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. OBJECTIVE CONVERGENCE
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    length_history,
    label="Path Length",
    linewidth=2
)

plt.plot(
    smooth_history,
    label="Path Smoothness",
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Objective value")

plt.title(
    "Convergence of the Two Research Objectives"
)

plt.grid(True)
plt.legend()

plt.show()