import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import random
import math
import ast
from haversine import haversine
from sklearn.cluster import KMeans
import elkai
import os

# -----------------------------
# GLOBAL PARAMETERS & RUN SETTINGS
# -----------------------------
speed_km_per_hr = 35
service_time_hr = 0.05
tmax = 3
hiring_cost_per_cluster = 50
distance_cost_per_km = 2

# Initial run parameters
max_merge_attempts_per_cluster = 20
max_iterations = 100
FIRST_RUN_NUM_RUNS = 2

# Second run parameters (to update after first round)
SECOND_RUN_MAX_MERGE_ATTEMPTS = 30
SECOND_RUN_MAX_ITERATIONS = 150
SECOND_RUN_NUM_RUNS = 2
# Define the depot location (latitude, longitude). Change to your actual depot location.
DEPOT = [41.0, 28.8]

# -----------------------------
# SPLIT–MERGE CLUSTERING CLASSES & FUNCTIONS
# -----------------------------
class Cluster:
    def __init__(self, data_points, cluster_id=None, color=None):
        # data_points: list (or np.array) of [lat, lon] points (does not include depot)
        self.data_points = np.array(data_points)
        self.centroid = self.calculate_centroid()
        self.id = cluster_id
        self.time = None
        self.cost = None
        self.nearest_cluster_ids = None
        self.tour = None      # for TSP route – will include depot as index 0
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
        distances = []
        for other in all_clusters:
            if other.id != self.id:
                d = haversine(self.centroid, other.centroid)
                distances.append((d, other.id))
        distances.sort(key=lambda x: x[0])
        self.nearest_cluster_ids = [distances[i][1] for i in range(min(num_nearest, len(distances)))]

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
        labels = kmeans.fit_predict(self.data_points)
        sub_points = [self.data_points[labels == i] for i in range(n_clusters)]
        subs = [Cluster(points, cluster_id=f"{self.id}_Sub{i+1}", color=self.color) 
                for i, points in enumerate(sub_points) if len(points) > 0]
        return subs

    def __repr__(self):
        return f"Cluster(ID={self.id}, Centroid={self.centroid.round(2)}, Points={len(self.data_points)}, Time={self.time}, Cost={self.cost})"

def calculate_total_distance_cluster_obj(cluster_obj, tour):
    pts = cluster_obj.data_points
    total_distance = 0.0
    if tour:
        for i in range(len(tour) - 1):
            total_distance += haversine(pts[tour[i]], pts[tour[i+1]])
    return total_distance

def calculate_total_time_cluster_obj(cluster_obj, tour, speed_km_per_hr, service_time_hr):
    total_distance = calculate_total_distance_cluster_obj(cluster_obj, tour)
    travel_time = total_distance / speed_km_per_hr
    total_time = travel_time + len(cluster_obj.data_points) * service_time_hr
    return total_time

def calculate_total_cost_cluster_obj(cluster_obj, total_distance, hiring_cost_per_cluster, distance_cost_per_km):
    return hiring_cost_per_cluster + (total_distance * distance_cost_per_km)

def solve_tsp_elkai_constrained_cluster_obj(cluster_obj, tmax, speed_km_per_hr, service_time_hr):
    """
    Solves a TSP that starts and ends at the depot.
    Given cluster_obj.data_points (the cluster's points), we prepend the DEPOT.
    The returned tour is in the extended space: index 0 = depot, 1... = cluster points.
    """
    pts = cluster_obj.data_points
    if len(pts) == 0:
        return None, None, None
    # Build list: depot first, then cluster points
    all_points = [DEPOT] + pts.tolist()
    n = len(all_points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine(all_points[i], all_points[j])
    dist_matrix_int = np.round(dist_matrix * 1000).astype(int)
    try:
        route = elkai.solve_int_matrix(dist_matrix_int)
    except RuntimeError:
        return None, None, None
    # Compute the total distance using dist_matrix
    total_dist = 0.0
    for i in range(len(route)-1):
        total_dist += dist_matrix[route[i], route[i+1]]
    total_dist += dist_matrix[route[-1], route[0]]  # close loop

    total_time = total_dist / speed_km_per_hr + len(pts) * service_time_hr
    if total_time <= tmax:
        return route, total_dist, total_time
    else:
        return None, total_dist, total_time

# (The merge, split, and iterative optimization functions remain unchanged.)
def attempt_cluster_merge(base_cluster, merge_candidate_cluster, all_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    if (base_cluster.time is not None and merge_candidate_cluster.time is not None) and (base_cluster.time + merge_candidate_cluster.time) > tmax:
        return False
    merged = np.concatenate((base_cluster.data_points, merge_candidate_cluster.data_points), axis=0)
    temp_cluster = Cluster(merged, cluster_id=f"TEMP_{base_cluster.id}_{merge_candidate_cluster.id}")
    tour_res, dist_res, time_res = solve_tsp_elkai_constrained_cluster_obj(temp_cluster, tmax, speed_km_per_hr, service_time_hr)
    if tour_res and time_res <= tmax:
        base_cluster.data_points = merged
        base_cluster.update_centroid()
        base_cluster.set_tsp_result(tour_res, dist_res)
        base_cluster.set_time(time_res)
        base_cluster.set_cost(calculate_total_cost_cluster_obj(base_cluster, dist_res, hiring_cost_per_cluster, distance_cost_per_km))
        return True
    else:
        return False

def explore_split_branches(current_clusters, sub_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    best_branch = copy.deepcopy(current_clusters)
    min_cost = sum(c.cost for c in get_valid_clusters(best_branch, tmax))
    def recursive_merge(clusters_to_merge, remaining_sub_clusters):
        nonlocal min_cost, best_branch
        if not remaining_sub_clusters:
            temp = copy.deepcopy(clusters_to_merge)
            temp = run_merging_iterations(temp, 1, speed_km_per_hr, service_time_hr, tmax, history_log)
            valid = get_valid_clusters(temp, tmax)
            if valid:
                current_cost = sum(c.cost for c in valid)
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_branch = copy.deepcopy(temp)
            return
        first = remaining_sub_clusters[0]
        rest = remaining_sub_clusters[1:]
        branch1 = copy.deepcopy(clusters_to_merge)
        branch1.append(first)
        recursive_merge(branch1, rest)
        branch2 = copy.deepcopy(clusters_to_merge)
        branch2.append(first)
        recursive_merge(branch2, rest)
    recursive_merge(copy.deepcopy(current_clusters), sub_clusters)
    return best_branch

def attempt_split_merge(base_cluster, all_clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
    original = np.copy(base_cluster.data_points)
    subs = base_cluster.split_cluster_kmeans(n_clusters=2)
    if not subs or len(subs) < 2:
        return False
    before = copy.deepcopy(all_clusters)
    best_branch = explore_split_branches(before, subs, speed_km_per_hr, service_time_hr, tmax, history_log)
    cost_before = sum(c.cost for c in get_valid_clusters(before, tmax))
    cost_after = sum(c.cost for c in get_valid_clusters(best_branch, tmax))
    if cost_after < cost_before:
        all_clusters.clear()
        all_clusters.extend(best_branch)
        return True
    else:
        base_cluster.data_points = original
        base_cluster.update_centroid()
        return False

def run_merging_iterations(clusters, iteration_count, speed_km_per_hr, service_time_hr, tmax, history_log):
    merged_indices = []
    random.shuffle(clusters)
    idx = 0
    while idx < len(clusters):
        base = clusters[idx]
        if base in merged_indices:
            idx += 1
            continue
        if base.merge_attempts_remaining <= 0 or base.get_attempts_left() <= 0:
            idx += 1
            continue
        base.calculate_nearest_clusters(clusters)
        nearest = base.nearest_cluster_ids or []
        attempted = False
        for n_id in nearest[:3]:
            candidate = next((c for c in clusters if c.id == n_id), None)
            if candidate and candidate not in merged_indices:
                if attempt_cluster_merge(base, candidate, clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
                    merged_indices.append(candidate)
                    base.decrement_merge_attempts()
                    base.decrement_attempts_left()
                    candidate.decrement_attempts_left()
                    attempted = True
                    break
                else:
                    base.decrement_merge_attempts()
                    base.decrement_attempts_left()
                    candidate.decrement_attempts_left()
                    attempted = True
        if (not attempted) and len(base.data_points) > 1 and base.get_attempts_left() > 0:
            if attempt_split_merge(base, clusters, speed_km_per_hr, service_time_hr, tmax, history_log):
                base.decrement_attempts_left()
            else:
                base.decrement_attempts_left()
        idx += 1
    valid = [c for c in clusters if c not in merged_indices]
    return valid

def get_valid_clusters(clusters, tmax):
    return [c for c in clusters if c.time is not None and c.time <= tmax]

def optimize_clustering(cluster_points_list, run_id):
    clusters_created = []
    initial_clusters = []
    cluster_colors = cm.viridis(np.linspace(0, 1, len(cluster_points_list)))
    history_log = []
    for i, points in enumerate(cluster_points_list):
        cluster = Cluster(points, cluster_id=f"Run{run_id}_Cluster_{i+1}", color=cluster_colors[i])
        clusters_created.append(cluster)
        initial_clusters.append(cluster)
    for cluster in clusters_created:
        tour, dist, time_val = solve_tsp_elkai_constrained_cluster_obj(cluster, tmax, speed_km_per_hr, service_time_hr)
        if tour:
            cluster.set_tsp_result(tour, dist)
            cluster.set_time(time_val)
            cluster.set_cost(calculate_total_cost_cluster_obj(cluster, dist, hiring_cost_per_cluster, distance_cost_per_km))
    initial_valid = get_valid_clusters(clusters_created, tmax)
    current = list(initial_valid)
    for iter in range(max_iterations):
        current = run_merging_iterations(current, 1, speed_km_per_hr, service_time_hr, tmax, history_log)
    final_valid = get_valid_clusters(current, tmax)
    total_cost = sum(c.cost for c in final_valid)
    total_time = sum(c.time for c in final_valid)
    initial_metrics = (len(initial_valid), sum(c.cost for c in initial_valid), sum(c.time for c in initial_valid))
    final_metrics = (len(final_valid), total_cost, total_time)
    return initial_metrics, final_metrics, final_valid

# -----------------------------
# UPDATED PLOTTING FUNCTIONS
# -----------------------------
def plot_input_clusters(csv_groups, filename=None):
    """
    Plots the raw input clusters from the CSV.
    Each row is plotted in a distinct color.
    """
    plt.figure(figsize=(12, 10))
    cmap = cm.get_cmap("tab20", len(csv_groups))
    for i, group in enumerate(csv_groups):
        arr = np.array(group)
        if arr.size == 0:
            continue
        color = cmap(i)
        plt.scatter(arr[:, 1], arr[:, 0], color=color, s=80, alpha=0.8, zorder=3, label=f"Group {i+1}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Input Clusters from CSV (Distinct Colors)")
    plt.axis("equal")
    plt.legend(loc='best')
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

def plot_final_clusters_with_routes(clusters, filename=None):
    """
    Plots the final clusters with optimized TSP routes.
    Each route starts at the depot (displayed as a red star).
    The cluster's tour (if available) is drawn as arrows.
    """
    plt.figure(figsize=(12, 10))
    cmap = cm.get_cmap("tab20", len(clusters))
    # Plot the depot as a red star.
    plt.plot(DEPOT[1], DEPOT[0], marker="*", color="red", markersize=15, zorder=5, label="Depot")
    for idx, cluster in enumerate(clusters):
        color = cmap(idx)
        pts = cluster.data_points
        # For TSP solution, we have an extended route:
        # route: indices in [0, 1, 2, ...] where 0 corresponds to DEPOT, and indices 1.. correspond to pts.
        if cluster.tour is not None and len(cluster.tour) > 1:
            # Create full list: depot + cluster points.
            all_points = np.vstack(([DEPOT], pts))
            # Scatter the cluster points in its color.
            plt.scatter(pts[:, 1], pts[:, 0], color=color, s=80, zorder=3, label=cluster.id)
            # Draw the route as arrows.
            route = cluster.tour
            for i in range(len(route) - 1):
                start = all_points[route[i]]
                end = all_points[route[i+1]]
                plt.annotate("",
                             xy=(end[1], end[0]), xycoords="data",
                             xytext=(start[1], start[0]), textcoords="data",
                             arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkA=5, shrinkB=5),
                             zorder=2)
            if route[-1] != route[0]:
                start = all_points[route[-1]]
                end = all_points[route[0]]
                plt.annotate("",
                             xy=(end[1], end[0]), xycoords="data",
                             xytext=(start[1], start[0]), textcoords="data",
                             arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkA=5, shrinkB=5),
                             zorder=2)
        else:
            if pts.size > 0:
                plt.scatter(pts[:, 1], pts[:, 0], color=color, s=80, zorder=3, label=cluster.id)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Final Clusters with Optimized Routes (Depot Marked)")
    plt.axis("equal")
    plt.legend(loc="best")
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

# -----------------------------
# UTILITY FUNCTION FOR COST DETAILS & CSV OUTPUT
# -----------------------------
def compute_cost_details(final_clusters):
    """
    Returns a DataFrame with columns:
    Cluster ID, # Nodes, Cost ($), Time (hrs), Distance (km).
    """
    data = []
    overall_cost = 0.0
    overall_time = 0.0
    for cluster in final_clusters:
        cost_val = cluster.cost if cluster.cost is not None else 0
        overall_cost += cost_val
        overall_time += cluster.time if cluster.time is not None else 0
        num_nodes = len(cluster.data_points)
        distance = cluster.total_distance if cluster.total_distance is not None else 0
        data.append({
            "Cluster ID": cluster.id,
            "# Nodes": num_nodes,
            "Cost ($)": round(cost_val, 2),
            "Time (hrs)": round(cluster.time, 2) if cluster.time is not None else None,
            "Distance (km)": round(distance, 2)
        })
    df = pd.DataFrame(data)
    return df, overall_cost, overall_time

def write_final_clusters_csv(final_clusters, part_number):
    nested_list = []
    for cluster in final_clusters:
        nested_list.append(str(cluster.data_points.tolist()))
    df = pd.DataFrame({"Final Cluster Coordinates": nested_list})
    csv_filename = f"split_merge_summary_part{part_number}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename

# -----------------------------
# PROCESSING SINGLE INPUT CSV FILE WITH TWO ITERATIONS
# -----------------------------
def process_input_file_split_merge(input_csv, part_number):
    df_in = pd.read_csv(input_csv)
    if df_in.columns[0].strip() == "Final Cluster Coordinates":
        coordinate_strings = df_in["Final Cluster Coordinates"].tolist()
        coordinates = [ast.literal_eval(cell) for cell in coordinate_strings if isinstance(cell, str)]
    else:
        coordinates = df_in.values.tolist()
    if not coordinates:
        print(f"No valid coordinates found in {input_csv}.")
        return {}
    # Use all rows from CSV (each row is a group of points)
    input_plot_file = f"input_clusters_part{part_number}.png"
    plot_input_clusters(coordinates, filename=input_plot_file)
    cluster_points_list = coordinates

    # ----- FIRST STEP (First Run) -----
    best_final_cost_first = float("inf")
    best_run_metrics_first = None
    best_run_clusters_first = None
    all_runs_initial_metrics_first = []
    for run_number in range(FIRST_RUN_NUM_RUNS):
        initial_metrics, final_metrics, final_clusters = optimize_clustering(cluster_points_list, run_number + 1)
        all_runs_initial_metrics_first.append(initial_metrics)
        print(f"\n=== FIRST STEP Run {run_number + 1} for file {input_csv} ===")
        print(f"  INITIAL Clusters: {initial_metrics[0]}, Cost: ${initial_metrics[1]:.2f}, Time: {initial_metrics[2]:.2f} hrs")
        print(f"  FINAL Clusters: {final_metrics[0]}, Cost: ${final_metrics[1]:.2f}, Time: {final_metrics[2]:.2f} hrs")
        if final_metrics[1] < best_final_cost_first:
            best_final_cost_first = final_metrics[1]
            best_run_metrics_first = final_metrics
            best_run_clusters_first = final_clusters

    # ----- SECOND STEP (Second Run) -----
    if best_run_clusters_first:
        final_cluster_points = [cluster.data_points.tolist() for cluster in best_run_clusters_first]
        print(f"\n--- Starting Second Step for file {input_csv} using best clusters from First Step ---")
        global max_merge_attempts_per_cluster, max_iterations
        orig_merge = max_merge_attempts_per_cluster
        orig_iter = max_iterations
        max_merge_attempts_per_cluster = SECOND_RUN_MAX_MERGE_ATTEMPTS
        max_iterations = SECOND_RUN_MAX_ITERATIONS

        best_final_cost_second = float("inf")
        best_run_metrics_second = None
        best_run_clusters_second = None
        all_runs_initial_metrics_second = []
        for run_number in range(SECOND_RUN_NUM_RUNS):
            initial_metrics2, final_metrics2, final_clusters2 = optimize_clustering(final_cluster_points, run_number + 100)
            all_runs_initial_metrics_second.append(initial_metrics2)
            print(f"\n=== SECOND STEP Run {run_number + 1} for file {input_csv} ===")
            print(f"  INITIAL Clusters: {initial_metrics2[0]}, Cost: ${initial_metrics2[1]:.2f}, Time: {initial_metrics2[2]:.2f} hrs")
            print(f"  FINAL Clusters: {final_metrics2[0]}, Cost: ${final_metrics2[1]:.2f}, Time: {final_metrics2[2]:.2f} hrs")
            if final_metrics2[1] < best_final_cost_second:
                best_final_cost_second = final_metrics2[1]
                best_run_metrics_second = final_metrics2
                best_run_clusters_second = final_clusters2

        second_plot_file = f"final_clusters_with_routes_part{part_number}.png"
        if best_run_clusters_second:
            plot_final_clusters_with_routes(best_run_clusters_second, filename=second_plot_file)
            cost_df_second, overall_cost_second, overall_time_second = compute_cost_details(best_run_clusters_second)
        else:
            cost_df_second = pd.DataFrame()
            overall_cost_second = 0
            overall_time_second = 0
            second_plot_file = ""
        max_merge_attempts_per_cluster = orig_merge
        max_iterations = orig_iter
    else:
        best_run_metrics_second = None
        best_run_clusters_second = None
        cost_df_second = pd.DataFrame()
        overall_cost_second = 0
        overall_time_second = 0
        second_plot_file = ""

    final_clusters_to_write = best_run_clusters_second if best_run_clusters_second else best_run_clusters_first
    final_clusters_csv = write_final_clusters_csv(final_clusters_to_write, part_number) if final_clusters_to_write else ""
    
    summary = {
        "part_number": part_number,
        "input_csv": input_csv,
        "input_plot_file": input_plot_file,
        "first_step": {
            "num_final_clusters": best_run_metrics_first[0] if best_run_metrics_first else 0,
            "overall_cost": best_run_metrics_first[1] if best_run_metrics_first else 0,
            "overall_time": best_run_metrics_first[2] if best_run_metrics_first else 0
        },
        "second_step": {
            "num_final_clusters": best_run_metrics_second[0] if best_run_metrics_second else (best_run_metrics_first[0] if best_run_metrics_first else 0),
            "overall_cost": overall_cost_second if best_run_clusters_second else (best_run_metrics_first[1] if best_run_metrics_first else 0),
            "overall_time": overall_time_second if best_run_clusters_second else (best_run_metrics_first[2] if best_run_metrics_first else 0),
            "cost_df": cost_df_second,
            "clusters_plot_file": second_plot_file
        },
        "cost_improvement": (best_run_metrics_first[1] - overall_cost_second) if best_run_clusters_second and best_run_metrics_first else 0,
        "final_clusters_csv": final_clusters_csv,
        "final_clusters": final_clusters_to_write
    }
    return summary

# -----------------------------
# MAIN FUNCTION: PROCESS MULTIPLE CSV FILES AND WRITE EXCEL SUMMARY
# -----------------------------
def main_multi_2(input_csv_files):
    summaries = []
    for idx, csv_file in enumerate(input_csv_files, start=1):
        print(f"\nProcessing file: {csv_file}")
        summary = process_input_file_split_merge(csv_file, part_number=idx)
        if summary:
            summaries.append(summary)
    excel_filename = "split_merge_summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
        for summary in summaries:
            sheet_name = f"PART {summary['part_number']}"
            overall_info = pd.DataFrame({
                "Field": [
                    "Input CSV",
                    "First Step - # Clusters",
                    "First Step - Overall Cost ($)",
                    "First Step - Overall Time (hrs)",
                    "Second Step - # Clusters",
                    "Second Step - Overall Cost ($)",
                    "Second Step - Overall Time (hrs)",
                    "Cost Improvement ($)"
                ],
                "Value": [
                    summary["input_csv"],
                    summary["first_step"]["num_final_clusters"],
                    summary["first_step"]["overall_cost"],
                    summary["first_step"]["overall_time"],
                    summary["second_step"]["num_final_clusters"],
                    summary["second_step"]["overall_cost"],
                    summary["second_step"]["overall_time"],
                    summary["cost_improvement"]
                ]
            })
            overall_info.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            if summary.get("final_clusters"):
                final_df, _, _ = compute_cost_details(summary["final_clusters"])
            else:
                final_df = pd.DataFrame()
            final_df.to_excel(writer, sheet_name=sheet_name, startrow=10, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.write("A8", "Overall Summary")
            worksheet.write("A20", "Input Clusters Plot")
            worksheet.insert_image("A21", summary["input_plot_file"], {"x_scale": 0.7, "y_scale": 0.7})
            worksheet.write("H20", "Final Clusters with Routes (Second Step)")
            if summary["second_step"]["clusters_plot_file"]:
                worksheet.insert_image("H21", summary["second_step"]["clusters_plot_file"], {"x_scale": 0.7, "y_scale": 0.7})
            worksheet.write("H40", "Final Clusters CSV File")
            worksheet.write("H41", summary["final_clusters_csv"])
    print(f"\nExcel summary file written: {excel_filename}")

# -----------------------------
# MAIN BLOCK
# -----------------------------
if __name__ == "__main__":
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
    ]
    main_multi_2(input_csv_files)
