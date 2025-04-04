import numpy as np
from math import radians, sin, cos, sqrt, atan2
import random
import elkai
from sklearn.cluster import KMeans
from typing import List, Tuple
import gurobipy as gp

class ClusterManager:
    def __init__(self, coordinates, k):
        self.coordinates = coordinates
        self.k = k
        self.clusters = {}  # Stores clusters by method name
        self.cluster_methods = {}  # Maps method names to clustering functions

    def add_clustering_method(self, method_name, clustering_function):
        """Add a clustering method to the manager."""
        self.cluster_methods[method_name] = clustering_function

    def run_clustering(self, method_name, **kwargs):
        """Run a specific clustering method and store the results."""
        if method_name not in self.cluster_methods:
            raise ValueError(f"Clustering method '{method_name}' not found.")

        clustering_function = self.cluster_methods[method_name]
        clusters = clustering_function(self.coordinates, **kwargs)
        self.clusters[method_name] = {f"{method_name}_cluster_{i+1}": cluster for i, cluster in enumerate(clusters)}

    def calculate_route_time(self, cluster_points, speed_km_per_hr, service_time_hr):
        """Calculate the total time for a given cluster."""
        depot = cluster_points[0]
        total_time = calculate_cluster_time(cluster_points, depot, speed_km_per_hr, service_time_hr)
        return total_time

    def get_all_clusters_with_methods(self, speed_km_per_hr, service_time_hr):
        """Retrieve all clusters with their methods, IDs, points, and costs."""
        all_clusters = []
        for method_name, clusters in self.clusters.items():
            for cluster_id, cluster_points in clusters.items():
                all_clusters.append({
                    "method": method_name,
                    "cluster_id": cluster_id,
                    "points": cluster_points,
                    "cost": self.calculate_route_time(cluster_points, speed_km_per_hr, service_time_hr)
                })
        return all_clusters

# Haversine distance function
def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calculate_cluster_time(cluster: List[Tuple[float, float]], depot: Tuple[float, float], speed_km_per_hr: float, service_time_hr: float) -> float:
    if len(cluster) <= 2:
        # If the cluster is too small, just calculate the distance between the points
        total_distance = haversine(cluster[0], cluster[1]) if len(cluster) == 2 else 0.0
        travel_time = total_distance / speed_km_per_hr
        total_service_time = (len(cluster) - 1) * service_time_hr  # Exclude depot
        total_time = travel_time + total_service_time
        return total_time

    distance_matrix = []
    for i in range(len(cluster)):
        row = []
        for j in range(len(cluster)):
            row.append(haversine(cluster[i], cluster[j]))
        distance_matrix.append(row)

    # Convert distance matrix to integers for elkai
    int_distance_matrix = [[int(dist * 1000) for dist in row] for row in distance_matrix]

    route = elkai.solve_int_matrix(int_distance_matrix)
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]

    travel_time = total_distance / speed_km_per_hr
    total_service_time = (len(cluster) - 1) * service_time_hr  # Exclude depot
    total_time = travel_time + total_service_time
    return total_time

# Nearest neighbor clustering with time constraint
def nearest_neighbor_clustering(coords: List[Tuple[float, float]], speed_km_per_hr: float, service_time_hr: float, tmax: float) -> List[List[Tuple[float, float]]]:
    depot = coords[0]
    coords = coords[1:]
    n = len(coords)
    clusters = []
    visited = set()

    while len(visited) < n:
        unvisited = [i for i in range(n) if i not in visited]
        if not unvisited:
            break

        start_point = random.choice(unvisited)
        cluster = [depot, coords[start_point]]
        visited.add(start_point)

        while True:
            nearest_point = None
            min_distance = float('inf')

            for i in range(n):
                if i not in visited:
                    distance = haversine(cluster[-1], coords[i])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = i

            if nearest_point is not None:
                temp_cluster = cluster + [coords[nearest_point]]
                total_time = calculate_cluster_time(temp_cluster, depot, speed_km_per_hr, service_time_hr)

                if total_time <= tmax:
                    cluster.append(coords[nearest_point])
                    visited.add(nearest_point)
                else:
                    break
            else:
                break

        cluster.append(depot)
        clusters.append(cluster)

    return clusters

def randomized_nearest_neighbor_clustering(coords: List[Tuple[float, float]], speed_km_per_hr: float, service_time_hr: float, tmax: float, k: int = 3) -> List[List[Tuple[float, float]]]:
    depot = coords[0]
    coords = coords[1:]
    n = len(coords)
    clusters = []
    visited = set()

    while len(visited) < n:
        unvisited = [i for i in range(n) if i not in visited]
        if not unvisited:
            break

        start_point = random.choice(unvisited)
        cluster = [depot, coords[start_point]]
        visited.add(start_point)

        while True:
            distances = []
            for i in range(n):
                if i not in visited:
                    distance = haversine(cluster[-1], coords[i])
                    distances.append((distance, i))

            if not distances:
                break

            distances.sort()
            k_nearest = [i for (_, i) in distances[:k]]
            nearest_point = random.choice(k_nearest)

            temp_cluster = cluster + [coords[nearest_point]]
            total_time = calculate_cluster_time(temp_cluster, depot, speed_km_per_hr, service_time_hr)

            if total_time <= tmax:
                cluster.append(coords[nearest_point])
                visited.add(nearest_point)
            else:
                break

        cluster.append(depot)
        clusters.append(cluster)

    return clusters

def recluster_problematic_clusters(clusters: List[List[Tuple[float, float]]], depot: Tuple[float, float], speed_km_per_hr: float, service_time_hr: float, tmax: float) -> List[List[Tuple[float, float]]]:
    new_clusters = []
    for cluster in clusters:
        total_time = calculate_cluster_time(cluster, depot, speed_km_per_hr, service_time_hr)

        if total_time <= tmax:
            new_clusters.append(cluster)
        else:
            kmeans = KMeans(n_clusters=2, random_state=42)
            coords = np.array(cluster)
            labels = kmeans.fit_predict(coords)

            cluster1 = coords[labels == 0].tolist()
            cluster2 = coords[labels == 1].tolist()

            cluster1.insert(0, depot)
            cluster2.insert(0, depot)

            reclustered = recluster_problematic_clusters([cluster1, cluster2], depot, speed_km_per_hr, service_time_hr, tmax)
            new_clusters.extend(reclustered)

    return new_clusters

def kmeans_clustering_with_constraint(
    customers: List[Tuple[float, float]],
    depot: Tuple[float, float],
    speed_km_per_hr: float,
    service_time_hr: float,
    tmax: float,
    initial_k: int = 3
) -> List[List[Tuple[float, float]]]:

    # Edge case
    if not customers:
        return []

    n = len(customers)
    k = min(initial_k, n)
    coords = np.array(customers)

    # 1) Run KMeans on customers only
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(coords)

    # 2) Build tours and recursively split oversized ones
    final_clusters = []
    for i in range(k):
        cluster_customers = coords[labels == i].tolist()
        if not cluster_customers:
            continue

        tour = [depot] + cluster_customers + [depot]
        final_clusters.extend(
            _recluster_if_needed(tour, depot, speed_km_per_hr, service_time_hr, tmax)
        )

    return final_clusters
def enforce_unique_assignment(
    tours: List[List[Tuple[float, float]]],
    depot: Tuple[float, float],
    speed_km_per_hr: float,
    service_time_hr: float,
    tmax: float
) -> List[List[Tuple[float, float]]]:
    point_to_tours = {}
    for idx, tour in enumerate(tours):
        for cust in tour[1:-1]:
            point_to_tours.setdefault(tuple(cust), []).append(idx)

    tours = [list(t) for t in tours]

    for pt, idxs in point_to_tours.items():
        if len(idxs) <= 1:
            continue

        # Remove pt from every candidate tour
        for i in idxs:
            tours[i] = [depot] + [c for c in tours[i][1:-1] if tuple(c) != pt] + [depot]

        # Re-insert into the single best tour
        best_idx, best_diff = None, float('inf')
        for i in idxs:
            candidate = tours[i][1:-1] + [list(pt)]
            cost = calculate_cluster_time([depot] + candidate + [depot], depot, speed_km_per_hr, service_time_hr)
            diff = abs(tmax - cost)
            if diff < best_diff:
                best_diff, best_idx = diff, i

        tours[best_idx] = [depot] + tours[best_idx][1:-1] + [list(pt)] + [depot]

    return [tour for tour in tours if len(tour) > 2]


def _recluster_if_needed(
    tour: List[Tuple[float, float]],
    depot: Tuple[float, float],
    speed: float,
    service: float,
    tmax: float
) -> List[List[Tuple[float, float]]]:
    """Split tour into two if it violates tmax, otherwise return it."""
    total_time = calculate_cluster_time(tour, depot, speed, service)
    if total_time <= tmax or len(tour) <= 3:
        return [tour]

    # Remove depot endpoints for splitting
    customers = tour[1:-1]
    coords = np.array(customers)
    labels = KMeans(n_clusters=2, random_state=42).fit_predict(coords)

    clusters = []
    for label in (0, 1):
        subset = [customers[i] for i in range(len(customers)) if labels[i] == label]
        new_tour = [depot] + subset + [depot]
        clusters.extend(_recluster_if_needed(new_tour, depot, speed, service, tmax))

    return clusters

# Example usage
coordinates = [[41.0618569, 28.6878197], [41.0696014, 28.8039982], [41.0797386, 28.8032557], [41.0860443, 28.8041662], [41.083317, 28.7821641], [41.0625786, 28.6854147], [41.1085064, 28.8095854], [41.105383, 28.6691296], [41.0738958, 28.7509008], [41.1200864, 28.8012589], [41.0680676, 28.6903563], [41.1019435, 28.798709], [41.0611008, 28.6876303], [41.0601831, 28.6884886], [41.0600444, 28.6880999], [41.0621453, 28.6875973], [41.062229, 28.8061252], [41.0655414, 28.7953399], [41.0623374, 28.8081159], [41.0635802, 28.7466293], [41.0643299, 28.7515568], [41.0666084, 28.8044928], [41.0981387, 28.7674465], [41.0845531, 28.7967313], [41.0802464, 28.7941205], [41.0800851, 28.7940823], [41.0666352, 28.8045071], [41.1229643, 28.7474467], [41.0670737, 28.8053667]] # Reduced for example
speed_km_per_hr = 35.0
service_time_hr = 0.05
tmax = 3.0

# Initialize ClusterManager
cluster_manager = ClusterManager(coordinates, k=3)

# Add clustering methods
cluster_manager.add_clustering_method("nearest_neighbor", lambda coords, **kwargs: nearest_neighbor_clustering(coords, speed_km_per_hr, service_time_hr, tmax))
cluster_manager.add_clustering_method("randomized_nearest_neighbor", lambda coords, **kwargs: randomized_nearest_neighbor_clustering(coords, speed_km_per_hr, service_time_hr, tmax))
cluster_manager.add_clustering_method("kmeans", lambda coords, **kwargs: kmeans_clustering_with_constraint(coords[1:], coords[0], speed_km_per_hr, service_time_hr, tmax))

# Run clustering methods
cluster_manager.run_clustering("nearest_neighbor")
cluster_manager.run_clustering("randomized_nearest_neighbor")
cluster_manager.run_clustering("kmeans")

# Retrieve all clusters with methods and costs
all_clusters = cluster_manager.get_all_clusters_with_methods(speed_km_per_hr, service_time_hr)

# Print results
for cluster in all_clusters:
    print(f"Method: {cluster['method']}, Cluster ID: {cluster['cluster_id']}, Cost: {cluster['cost']:.2f} hours")
    print(f"Points: {cluster['points']}")
    print()

def optimal_set_covering_gurobi(universe, sets):
    try:
        # Create a new Gurobi model
        model = gp.Model("set_covering")

        # Binary variables indicating if a set is selected
        x = model.addVars(len(sets), vtype=gp.GRB.BINARY, name="select")

        # Objective: Minimize the total cost of selected sets
        costs = [s['cost'] for s in sets]
        model.setObjective(gp.quicksum(costs[i] * x[i] for i in range(len(sets))), gp.GRB.MINIMIZE)

        # Constraints: Ensure every point in the universe is covered by at least one set
        point_indices = {tuple(point): idx for idx, point in enumerate(universe)}
        for point in universe:
            point_idx = point_indices[tuple(point)]
            covering_sets = []
            for i, s in enumerate(sets):
                if list(point) in s['points']:
                    covering_sets.append(x[i])
            if covering_sets:
                model.addConstr(gp.quicksum(covering_sets) >= 1, f"cover_{point_idx}")
            else:
                print(f"Warning: Point {point} is not covered by any set.")
                return None # Should not happen if all individual points are in some clusters

        # Solve the model
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            selected_sets = [sets[i] for i in range(len(sets)) if x[i].x > 0.5]
            return selected_sets
        else:
            print(f"Optimal solution not found. Gurobi status: {model.status}")
            return None

    except gp.GurobiError as e:
        print(f"Error reported by Gurobi: {e}")
        return None

# Extract universe
universe = []
for coord in coordinates:
    universe.append(coord)

# Call optimal set covering with Gurobi
optimal_clusters = optimal_set_covering_gurobi(universe, all_clusters)

# Output final result
# After solving set covering:
if optimal_clusters:
    depot = coordinates[0]
    selected_tours = [cluster['points'] for cluster in optimal_clusters]