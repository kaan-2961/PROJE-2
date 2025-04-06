import numpy as np
import elkai
#from haversine import haversine  # REMOVED: We will use our own haversine function.
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import copy
import random
import pandas as pd
import xlsxwriter
import ast
import math

# Define parameters
speed_km_per_hr = 35
service_time_hr = 0.05
tmax = 3
hiring_cost_per_cluster = 50
distance_cost_per_km = 2
max_merge_attempts_per_cluster = 20
max_iterations = 100

# NEW: Custom haversine function (in km)
def haversine(coord1, coord2):
    """
    Calculate the haversine distance (in km) between two [lat, lon] points.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Cluster class
class Cluster:
    def __init__(self, data_points, cluster_id=None, color=None):
        self.data_points = np.array(data_points)
        self.centroid = self.calculate_centroid()
        self.id = cluster_id
        self.time = None
        self.cost = None
        self.nearest_cluster_ids = None
        self.tour = None
        self.total_distance = None
        self.merge_attempts_remaining = max_merge_attempts_per_cluster
        self.attempts_left = 4
        if color is None:
            self.color = np.random.rand(3,)
        else:
            self.color = color

    def calculate_centroid(self):
        if not self.data_points.size:
            return np.array([0, 0])
        return np.mean(self.data_points, axis=0)

    def update_centroid(self):
        self.centroid = self.calculate_centroid()

    def set_time(self, time_value):
        self.time = time_value

    def set_cost(self, cost_value):
        self.cost = cost_value

    def set_tsp_result(self, tour, total_distance):
        self.tour = tour
        self.total_distance = total_distance

    def calculate_nearest_clusters(self, all_clusters, num_nearest=6):
        distances_to_other_clusters = []
        for other_cluster in all_clusters:
            if other_cluster.id != self.id:
                distance = haversine(self.centroid, other_cluster.centroid)
                distances_to_other_clusters.append((distance, other_cluster.id))
        distances_to_other_clusters.sort(key=lambda item: item[0])
        nearest_ids = [distances_to_other_clusters[i][1] for i in range(min(num_nearest, len(distances_to_other_clusters)))]
        self.nearest_cluster_ids = nearest_ids

    def decrement_merge_attempts(self):
        self.merge_attempts_remaining -= 1

    def decrement_attempts_left(self):
        self.attempts_left -= 1

    def get_attempts_left(self):
        return self.attempts_left

    def split_cluster_kmeans(self, n_clusters=2):
        if len(self.data_points) < n_clusters:
            return None
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(self.data_points)
        sub_clusters_points = [self.data_points[cluster_labels == i] for i in range(n_clusters)]
        sub_clusters = [Cluster(points, cluster_id=f"{self.id}_Sub{i+1}", color=self.color) for i, points in enumerate(sub_clusters_points) if len(points) > 0]
        return sub_clusters

    def __repr__(self):
        return f"Cluster(ID={self.id}, Centroid={self.centroid.round(2)}, Points={len(self.data_points)}, Time={self.time}, Cost={self.cost})"

# Utility functions
def calculate_total_distance_cluster_obj(cluster_obj, tour):
    cluster = cluster_obj.data_points
    total_distance = 0
    if tour:
        for i in range(len(tour) - 1):
            total_distance += haversine(cluster[tour[i]], cluster[tour[i + 1]])
    return total_distance

def calculate_total_time_cluster_obj(cluster_obj, tour, speed_km_per_hr, service_time_hr):
    total_distance = calculate_total_distance_cluster_obj(cluster_obj, tour)
    travel_time = total_distance / speed_km_per_hr
    total_time = travel_time + len(cluster_obj.data_points) * service_time_hr
    return total_time

# -------------------------------
# New Functions for Cost Calculation
# -------------------------------
def tsp_distance_final(points):
    """
    Compute a TSP-like route distance for a list of points using a simple nearest-neighbor heuristic.
    Assumes points is a list of [lat, lon] pairs.
    """
    if not points:
        return 0.0
    pts = points.copy()  # Copy to avoid modifying original list
    current = pts.pop(0)  # Start from the first point
    total_distance = 0.0
    while pts:
        next_point = min(pts, key=lambda p: haversine(current, p))
        total_distance += haversine(current, next_point)
        current = next_point
        pts.remove(next_point)
    return total_distance

def compute_cost_for_cluster(cluster_points):
    """
    Given a list of points (each a [lat, lon] pair) representing a final cluster,
    compute the TSP-like route distance, the distance cost, and the total cost.
    """
    route_distance = tsp_distance_final(cluster_points)
    distance_cost = route_distance * distance_cost_per_km
    total_cost = hiring_cost_per_cluster + distance_cost
    return route_distance, distance_cost, total_cost

def compute_cost_details(nested_list):
    """
    For each final cluster (list of coordinate pairs) in nested_list,
    compute:
      - Route Distance (km) using tsp_distance_final,
      - Distance Cost (route_distance * distance_cost_per_km),
      - Total Cluster Cost (hiring_cost_per_cluster + distance_cost).
    Returns a DataFrame and the overall total cost.
    """
    data = []
    overall_cost = 0.0
    for idx, cluster in enumerate(nested_list, start=1):
        route_dist, dist_cost, total_cost = compute_cost_for_cluster(cluster)
        overall_cost += total_cost
        data.append({
            'Cluster #': idx,
            'Num Points': len(cluster),
            'Route Distance (km)': round(route_dist, 2),
            'Distance Cost': round(dist_cost, 2),
            'Hiring Cost': hiring_cost_per_cluster,
            'Total Cost': round(total_cost, 2)
        })
    df = pd.DataFrame(data)
    return df, overall_cost

def calculate_total_cost_cluster_obj(cluster_obj, total_distance, hiring_cost_per_cluster, distance_cost_per_km):
    return hiring_cost_per_cluster + (total_distance * distance_cost_per_km)

def solve_tsp_elkai_constrained_cluster_obj(cluster_obj, tmax, speed_km_per_hr, service_time_hr):
    cluster = cluster_obj.data_points
    if len(cluster) < 2:
        return None, None, None
    if len(cluster) == 2:
        total_distance = haversine(cluster[0], cluster[1])
        total_time = total_distance / speed_km_per_hr + service_time_hr * 2
        tour = [0, 1]
        return tour, total_distance, total_time
    num_points = len(cluster)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = haversine(cluster[i], cluster[j])
    distance_matrix_int = np.round(distance_matrix * 1000).astype(int)
    try:
        tour = elkai.solve_int_matrix(distance_matrix_int)
    except RuntimeError:
        return None, None, None
    total_distance = calculate_total_distance_cluster_obj(cluster_obj, tour)
    total_time = calculate_total_time_cluster_obj(cluster_obj, tour, speed_km_per_hr, service_time_hr)
    if total_time <= tmax:
        return tour, total_distance, total_time
    else:
        return None, total_distance, total_time

def attempt_cluster_merge(base_cluster, merge_candidate_cluster, all_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    if (base_cluster.time is not None and merge_candidate_cluster.time is not None) and (base_cluster.time + merge_candidate_cluster.time) > tmax:
        return False
    merged_data_points = np.concatenate((base_cluster.data_points, merge_candidate_cluster.data_points), axis=0)
    temp_cluster = Cluster(merged_data_points, cluster_id=f"TEMP_MERGE_{base_cluster.id}_{merge_candidate_cluster.id}")
    tour_result, distance_result, time_result = solve_tsp_elkai_constrained_cluster_obj(temp_cluster, tmax, speed_km_per_hr, service_time_hr)
    if tour_result and time_result <= tmax:
        base_cluster.data_points = merged_data_points
        base_cluster.update_centroid()
        base_cluster.set_tsp_result(tour_result, distance_result)
        base_cluster.set_time(time_result)
        base_cluster.set_cost(calculate_total_cost_cluster_obj(base_cluster, distance_result, hiring_cost_per_cluster, distance_cost_per_km))
        return True
    else:
        return False

def explore_split_branches(current_clusters, sub_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    best_branch_clusters = copy.deepcopy(current_clusters)
    min_cost = sum(c.cost for c in get_valid_clusters(best_branch_clusters, tmax))

    def recursive_merge(clusters_to_merge, remaining_sub_clusters):
        nonlocal min_cost, best_branch_clusters

        if not remaining_sub_clusters:
            temp_clusters = copy.deepcopy(clusters_to_merge)
            temp_clusters = run_merging_iterations(temp_clusters, 1, speed_km_per_hr, service_time_hr, tmax, history_log)
            valid_clusters = get_valid_clusters(temp_clusters, tmax)
            if valid_clusters:
                current_cost = sum(c.cost for c in valid_clusters)
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_branch_clusters = copy.deepcopy(temp_clusters)
            return

        first_sub_cluster = remaining_sub_clusters[0]
        rest_sub_clusters = remaining_sub_clusters[1:]

        # Branch 1: Merge the first sub-cluster
        branch1_clusters = copy.deepcopy(clusters_to_merge)
        branch1_clusters.append(first_sub_cluster)
        recursive_merge(branch1_clusters, rest_sub_clusters)

        # Branch 2: Don't merge the first sub-cluster (keep it separate)
        branch2_clusters = copy.deepcopy(clusters_to_merge)
        branch2_clusters.append(first_sub_cluster)
        recursive_merge(branch2_clusters, rest_sub_clusters)

    recursive_merge(copy.deepcopy(current_clusters), sub_clusters)

    return best_branch_clusters

def attempt_split_merge(base_cluster, all_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    original_data_points = np.copy(base_cluster.data_points)
    sub_clusters = base_cluster.split_cluster_kmeans(n_clusters=2)
    if not sub_clusters or len(sub_clusters) < 2:
        return False
    initial_clusters_state_before_split = copy.deepcopy(all_clusters)
    best_branch_result_clusters = explore_split_branches(initial_clusters_state_before_split, sub_clusters, speed_km_per_hr, service_time_hr, tmax, history_log)
    initial_cost = sum(c.cost for c in get_valid_clusters(initial_clusters_state_before_split, tmax))
    final_cost_after_split_merge = sum(c.cost for c in get_valid_clusters(best_branch_result_clusters, tmax))
    if final_cost_after_split_merge < initial_cost:
        all_clusters.clear()
        all_clusters.extend(best_branch_result_clusters)
        return True
    else:
        base_cluster.data_points = original_data_points
        base_cluster.update_centroid()
        return False

def run_merging_iterations(clusters, iteration_count, speed_km_per_hr, service_time_hr, tmax, history_log):
    for iteration in range(iteration_count):
        clusters_merged_in_iteration = []
        random.shuffle(clusters)
        cluster_index = 0
        while cluster_index < len(clusters):
            base_cluster = clusters[cluster_index]
            if base_cluster in clusters_merged_in_iteration:
                cluster_index += 1
                continue
            if base_cluster.merge_attempts_remaining <= 0 or base_cluster.get_attempts_left() <= 0:
                cluster_index += 1
                continue
            base_cluster.calculate_nearest_clusters(clusters)
            nearest_cluster_ids = base_cluster.nearest_cluster_ids or []
            direct_merge_attempted_in_iteration = False
            for nearest_cluster_id in nearest_cluster_ids[:3]:
                merge_candidate_cluster = next((c for c in clusters if c.id == nearest_cluster_id), None)
                if merge_candidate_cluster and merge_candidate_cluster not in clusters_merged_in_iteration:
                    if attempt_cluster_merge(base_cluster, merge_candidate_cluster, clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
                        clusters_merged_in_iteration.append(merge_candidate_cluster)
                        base_cluster.decrement_merge_attempts()
                        base_cluster.decrement_attempts_left()
                        merge_candidate_cluster.decrement_attempts_left()
                        direct_merge_attempted_in_iteration = True
                        break
                    else:
                        base_cluster.decrement_merge_attempts()
                        base_cluster.decrement_attempts_left()
                        merge_candidate_cluster.decrement_attempts_left()
                        direct_merge_attempted_in_iteration = True
            if not direct_merge_attempted_in_iteration and len(base_cluster.data_points) > 1 and base_cluster.get_attempts_left() > 0:
                if attempt_split_merge(base_cluster, clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
                    base_cluster.decrement_attempts_left()
                else:
                    base_cluster.decrement_attempts_left()
            cluster_index += 1
        valid_clusters = [cluster for cluster in clusters if cluster not in clusters_merged_in_iteration]
        return valid_clusters

def get_valid_clusters(clusters, tmax):
    return [cluster for cluster in clusters if cluster.time is not None and cluster.time <= tmax]

# NEW: Added function to compute cost details from final clusters
def compute_cost_details_from_clusters(final_clusters):
    """
    Given a list of final Cluster objects, compute a summary DataFrame with:
      - Cluster ID
      - Number of points
      - Route Distance (km) (using pre-calculated total_distance if available, else recalculating)
      - Cost (using cluster.cost if available, else calculated via calculate_total_cost_cluster_obj)
      - Time (hrs)
    Also returns the overall total cost.
    """
    data = []
    overall_cost = 0.0
    for cluster in final_clusters:
        if cluster.total_distance is not None:
            route_distance = cluster.total_distance
        else:
            route_distance = calculate_total_distance_cluster_obj(cluster, cluster.tour) if cluster.tour else 0.0
        if cluster.cost is not None:
            cost = cluster.cost
        else:
            cost = calculate_total_cost_cluster_obj(cluster, route_distance, hiring_cost_per_cluster, distance_cost_per_km)
        overall_cost += cost
        data.append({
            "Cluster ID": cluster.id,
            "Num Points": len(cluster.data_points),
            "Route Distance (km)": round(route_distance, 2),
            "Cost": round(cost, 2),
            "Time (hrs)": round(cluster.time, 2) if cluster.time is not None else None
        })
    df = pd.DataFrame(data)
    return df, overall_cost

def optimize_clustering(cluster_points_list, run_id):
    clusters_created = []
    initial_clusters = []
    cluster_colors = cm.viridis(np.linspace(0, 1, len(cluster_points_list)))
    history_log = []
    for i, points in enumerate(cluster_points_list):
        cluster = Cluster(points, cluster_id=f"Run{run_id}_Cluster_{i+1}", color=cluster_colors[i])
        clusters_created.append(cluster)
        initial_clusters.append(cluster)
    for cluster_instance in clusters_created:
        tour_result, distance_result, time_result = solve_tsp_elkai_constrained_cluster_obj(cluster_instance, tmax, speed_km_per_hr, service_time_hr)
        if tour_result:
            cluster_instance.set_tsp_result(tour_result, distance_result)
            cluster_instance.set_time(time_result)
            cluster_instance.set_cost(calculate_total_cost_cluster_obj(cluster_instance, distance_result, hiring_cost_per_cluster, distance_cost_per_km))
    initial_valid_clusters = get_valid_clusters(clusters_created, tmax)
    current_clusters = list(initial_valid_clusters)
    for iteration in range(max_iterations):
        current_clusters = run_merging_iterations(current_clusters, 1, speed_km_per_hr, service_time_hr, tmax, history_log)
    final_valid_clusters = get_valid_clusters(current_clusters, tmax)
    final_cost = sum(c.cost for c in final_valid_clusters)
    final_time = sum(c.time for c in final_valid_clusters)
    initial_metrics = (len(initial_valid_clusters),
                       sum(c.cost for c in initial_valid_clusters),
                       sum(c.time for c in initial_valid_clusters))
    final_metrics = (len(final_valid_clusters), final_cost, final_time)
    # NEW: Compute cost details from final_valid_clusters.
    cost_df, overall_cost = compute_cost_details_from_clusters(final_valid_clusters)
    return initial_metrics, final_metrics, final_valid_clusters, cost_df, overall_cost

if __name__ == '__main__':
    # List your input CSV files (from your split–merge process)
    input_csv_files = [
        "split_merge_osm_data_part1.csv",
        "split_merge_osm_data_part2.csv",
        "split_merge_osm_data_part3.csv",
        "split_merge_osm_data_part4.csv",
        "split_merge_osm_data_part5.csv",
        "split_merge_osm_data_part6.csv",
        "split_merge_osm_data_part7.csv",
        "split_merge_osm_data_part8.csv",
        "split_merge_osm_data_part9.csv",
        "split_merge_osm_data_part10.csv",
        "split_merge_osm_data_part11.csv",
        "split_merge_osm_data_part12.csv",
        # ... add additional CSV file names as needed ...
    ]
    
    num_runs = 20  # Number of runs per input file
    
    # Loop over each input CSV file.
    for csv_file in input_csv_files:
        print(f"\nProcessing file: {csv_file}")
        # Read the CSV and convert the "Final Cluster Coordinates" column.
        df_in = pd.read_csv(csv_file)
        cluster_points_list = df_in["Final Cluster Coordinates"].apply(ast.literal_eval).tolist()
    
        best_final_cost = float('inf')
        best_run_metrics = None
        best_run_clusters = None
        best_run_cost_df = None
        all_runs_initial_metrics = []
    
        # Run optimization num_runs times on the clusters from this CSV.
        for run_number in range(num_runs):
            # Note: optimize_clustering returns 5 values:
            # initial_metrics, final_metrics, final_valid_clusters, cost_df, overall_cost
            initial_metrics, final_metrics, final_clusters, cost_df, overall_cost = optimize_clustering(cluster_points_list, run_number + 1)
            all_runs_initial_metrics.append(initial_metrics)
            print(f"\n=== RUN {run_number + 1} METRICS ===")
            print(f"INITIAL Clusters: {initial_metrics[0]}, Total Cost: ${initial_metrics[1]:.2f}, Total Time: {initial_metrics[2]:.2f} hrs")
            print(f"FINAL Clusters: {final_metrics[0]}, Total Cost: ${final_metrics[1]:.2f}, Total Time: {final_metrics[2]:.2f} hrs")
            print(f"Overall Cost from Cost DF: ${overall_cost:.2f}")
            if final_metrics[1] < best_final_cost:
                best_final_cost = final_metrics[1]
                best_run_metrics = final_metrics
                best_run_clusters = final_clusters
                best_run_cost_df = cost_df
    
        print("\n=== BEST RUN OVERALL for file", csv_file, "===")
        if best_run_metrics:
            print(f"Clusters: {best_run_metrics[0]}")
            print(f"Total Cost: ${best_run_metrics[1]:.2f}")
            print(f"Total Time: {best_run_metrics[2]:.2f} hrs")
            print("\n=== BEST RUN FINAL CLUSTER DETAILS ===")
            if best_run_clusters:
                for cluster in best_run_clusters:
                    print(f"\nCluster {cluster.id}:")
                    print(f"Points: {len(cluster.data_points)}")
                    print(f"Time: {cluster.time:.2f} hrs | Cost: ${cluster.cost:.2f}")
                    print(f"Centroid: ({cluster.centroid[0]:.4f}, {cluster.centroid[1]:.4f})")
    
            initial_metrics_avg = np.mean(all_runs_initial_metrics, axis=0)
            print("\n=== AVERAGE INITIAL METRICS ACROSS ALL RUNS ===")
            print(f"Clusters: {initial_metrics_avg[0]:.2f}")
            print(f"Total Cost: ${initial_metrics_avg[1]:.2f}")
            print(f"Total Time: {initial_metrics_avg[2]:.2f} hrs")
    
            initial_metrics_best_run = all_runs_initial_metrics[np.argmin(np.array(all_runs_initial_metrics)[:, 1])]
            print("\n=== INITIAL METRICS FOR BEST RUN ===")
            print(f"Clusters: {initial_metrics_best_run[0]}")
            print(f"Total Cost: ${initial_metrics_best_run[1]:.2f}")
            print(f"Total Time: {initial_metrics_best_run[2]:.2f} hrs")
    
            print("\n=== OPTIMIZATION SUMMARY FOR BEST RUN ===")
            print(f"Clusters Reduced: {initial_metrics_best_run[0] - best_run_metrics[0]}")
            print(f"Cost Savings: ${initial_metrics_best_run[1] - best_run_metrics[1]:.2f}")
            print(f"Time Change: {best_run_metrics[2] - initial_metrics_best_run[2]:.2f} hrs")
    
            # Second Run Implementation on the best run clusters
            if best_run_clusters:
                final_cluster_points = [cluster.data_points.tolist() for cluster in best_run_clusters]
    
                # Reset parameters for second run if needed (e.g., increase merge attempts and iterations)
                max_merge_attempts_per_cluster = 30
                max_iterations = 150
                # (Make sure to update the Cluster class if you want to change attempts_left behavior.)
    
                # Run second optimization (re-using optimize_clustering)
                initial_metrics2, final_metrics2, final_clusters2, cost_df2, overall_cost2 = optimize_clustering(final_cluster_points, num_runs + 1)
    
                print("\n=== SECOND RUN METRICS ===")
                print(f"INITIAL Clusters: {initial_metrics2[0]}, Total Cost: ${initial_metrics2[1]:.2f}, Total Time: {initial_metrics2[2]:.2f} hrs")
                print(f"FINAL Clusters: {final_metrics2[0]}, Total Cost: ${final_metrics2[1]:.2f}, Total Time: {final_metrics2[2]:.2f} hrs")
    
                print("\n=== SECOND RUN FINAL CLUSTER DETAILS ===")
                if final_clusters2:
                    for cluster in final_clusters2:
                        print(f"\nCluster {cluster.id}:")
                        print(f"Points: {len(cluster.data_points)}")
                        print(f"Time: {cluster.time:.2f} hrs | Cost: ${cluster.cost:.2f}")
                        print(f"Centroid: ({cluster.centroid[0]:.4f}, {cluster.centroid[1]:.4f})")
        else:
            print("No valid solution found in any run for file", csv_file)
