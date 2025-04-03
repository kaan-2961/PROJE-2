import numpy as np
import elkai
from haversine import haversine
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import copy
import random
import copy

# Define parameters
speed_km_per_hr = 35
service_time_hr = 0.05
tmax = 3
hiring_cost_per_cluster = 50
distance_cost_per_km = 2
max_merge_attempts_per_cluster = 20
max_iterations = 100


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
    return initial_metrics, final_metrics, final_valid_clusters

if __name__ == '__main__':
    cluster_points_list = [[[41.019734, 28.81971], [41.11845, 28.77374], [41.11968, 28.76789], [41.13167, 28.78009], [41.11739, 28.77048], [41.12238, 28.7703], [41.11923, 28.76654], [41.11711, 28.77347], [41.12275, 28.77961], [41.12251, 28.77141], [41.11902, 28.77281], [41.12395, 28.77161], [41.1227, 28.77899], [41.12066, 28.7683]], [[41.019734, 28.81971], [41.0405, 28.84485], [41.03921, 28.84449], [41.03086, 28.84659], [41.046, 28.85154], [41.05089, 28.84942], [41.04835, 28.85265], [41.03122, 28.85626], [41.02748, 28.85722], [41.02941, 28.85426], [41.03148, 28.85003], [41.04302, 28.84782], [41.04868, 28.85969], [41.05049, 28.85791], [41.04183, 28.86832], [41.05047, 28.85793], [41.0447, 28.86895], [41.04836, 28.85967], [41.03808, 28.85012], [41.04339, 28.86287], [41.03493, 28.8549], [41.03605, 28.86014], [41.03296, 28.86349], [41.03947, 28.85785], [41.03648, 28.85858], [41.03467, 28.86679], [41.03436, 28.86568], [41.04027, 28.8593], [41.03573, 28.85994], [41.03668, 28.85772], [41.03421, 28.85995], [41.05024, 28.8604], [41.05332, 28.85363], [41.05088, 28.86582]], [[41.019734, 28.81971], [41.00251, 28.78824], [41.0095, 28.80104], [41.0061, 28.79811], [41.01474, 28.79731], [41.00336, 28.79283], [41.0091, 28.79973], [41.00481, 28.79416], [41.00267, 28.78858], [41.00241, 28.79292], [41.01, 28.77869], [41.01752, 28.78528], [41.01749, 28.78498], [41.01366, 28.78738], [41.00998, 28.7747], [41.01749, 28.78521], [41.01147, 28.77735], [41.00023, 28.79721], [40.99939, 28.79783]], [[41.019734, 28.81971], [41.00678, 28.85301], [40.99758, 28.84846], [41.00853, 28.84895], [40.99942, 28.84662], [41.00864, 28.8496], [40.99917, 28.85226], [41.0108, 28.85005], [40.97926, 28.8551], [40.99791, 28.86503], [40.9963, 28.86331], [40.99417, 28.8474], [40.99977, 28.84421], [40.996, 28.84095], [40.99268, 28.84227], [40.99227, 28.84081], [40.99451, 28.84576], [40.99266, 28.8456], [40.99462, 28.8413], [40.99825, 28.85663], [41.00108, 28.85698], [41.00201, 28.86023], [41.0052, 28.85914], [41.00309, 28.85529], [40.99837, 28.85601], [40.99957, 28.85926], [41.00348, 28.85935], [40.99959, 28.85742], [40.99788, 28.85626], [40.98885, 28.84529], [40.98465, 28.84658], [40.9941, 28.85229], [40.98582, 28.84913], [40.99704, 28.85532], [40.99732, 28.85544], [40.99581, 28.85401], [40.99581, 28.85402], [40.99726, 28.85566], [40.99382, 28.85224]], [[41.019734, 28.81971], [41.05943, 28.79879], [41.0511, 28.79667], [41.05471, 28.79464], [41.047, 28.8004], [41.05481, 28.79922], [41.05732, 28.80775], [41.05702, 28.80761], [41.05296, 28.80591], [41.05771, 28.80125], [41.05284, 28.79818], [41.06106, 28.80147], [41.05203, 28.79875], [41.04828, 28.8008], [41.05029, 28.80319], [41.05404, 28.79501], [41.05557, 28.78605], [41.05492, 28.79224], [41.05458, 28.7911], [41.03997, 28.81039], [41.04759, 28.80988]], [[41.019734, 28.81971], [40.98635, 28.61674]], [[41.019734, 28.81971], [41.04208, 28.82851], [41.04506, 28.83714], [41.04448, 28.83297], [41.04364, 28.84111], [41.0549, 28.82542], [41.04814, 28.82217], [41.0544, 28.84569], [41.04434, 28.81931], [41.05632, 28.84585], [41.05419, 28.83518], [41.06019, 28.83569], [41.0517, 28.84262], [41.04873, 28.84436], [41.04775, 28.84593], [41.04919, 28.83479], [41.04896, 28.83851], [41.04917, 28.8384], [41.04509, 28.8221], [41.04513, 28.82206]], [[41.019734, 28.81971], [41.08429, 28.77111], [41.09484, 28.77081], [41.0951, 28.77448], [41.10249, 28.76225], [41.10081, 28.76186], [41.10135, 28.76213], [41.10159, 28.76215]], [[41.019734, 28.81971], [41.01078, 28.82474], [41.00199, 28.83423], [40.99878, 28.83053], [41.00859, 28.82193], [41.00227, 28.83577], [41.00714, 28.81988], [41.00881, 28.83086], [41.00262, 28.8305], [41.00576, 28.82782], [40.99799, 28.83085], [41.00297, 28.83599], [41.00817, 28.8329], [40.99968, 28.83045], [41.00571, 28.83097], [41.00317, 28.83063], [41.00213, 28.83393], [41.0077, 28.81984], [41.00279, 28.83604], [40.99854, 28.83082], [41.01119, 28.82867], [41.00512, 28.8441], [41.00567, 28.84485], [41.00515, 28.84431], [41.00471, 28.84345], [41.01776, 28.84158], [41.01485, 28.84199], [41.01433, 28.83542], [41.01456, 28.83531], [41.01604, 28.82458], [40.99913, 28.8387], [41.00317, 28.84295], [41.01183, 28.83889], [41.00821, 28.83881], [41.0084, 28.84147], [41.00304, 28.84291], [41.01148, 28.83453], [41.00671, 28.84161], [41.00982, 28.81493], [40.9853, 28.83361]], [[41.019734, 28.81971], [41.04149, 28.87133], [41.04187, 28.871], [41.03977, 28.87029], [41.02978, 28.86965], [41.03007, 28.86985], [41.03067, 28.87929], [41.03773, 28.87727], [41.02525, 28.87271], [41.03343, 28.88799], [41.03411, 28.88888], [41.03381, 28.88571], [41.0297, 28.89694], [41.03522, 28.88561], [41.02693, 28.88161], [41.03547, 28.87944], [41.0319, 28.87701], [41.04424, 28.87849], [41.04128, 28.88248], [41.04649, 28.87338], [41.04569, 28.88007], [41.0392, 28.87739], [41.0461, 28.87777], [41.04211, 28.87758], [41.04037, 28.88336], [41.04522, 28.87827], [41.04229, 28.87845], [41.02297, 28.87718], [41.01933, 28.88011], [41.01944, 28.89975], [41.02107, 28.87962], [41.02015, 28.87892], [41.01604, 28.89648]], [[41.019734, 28.81971], [40.99978, 28.8703], [41.00144, 28.874], [41.00447, 28.88763], [41.007, 28.88804], [40.97824, 28.87238], [40.9774, 28.87706], [40.97766, 28.87685], [40.97849, 28.87509], [40.97914, 28.87355], [40.97869, 28.87964], [40.97806, 28.87267], [40.97829, 28.87212], [40.97769, 28.8769], [40.97892, 28.87213], [40.97966, 28.87484], [40.98926, 28.87012], [40.98249, 28.87297], [40.99214, 28.86988], [40.98969, 28.86953], [40.98754, 28.8661], [40.98105, 28.87006], [40.9868, 28.86887], [40.99237, 28.88436], [40.99018, 28.87662], [40.98867, 28.87346], [40.99727, 28.86732], [40.98886, 28.87139], [40.99225, 28.882], [40.99365, 28.87432], [40.9848, 28.87811], [40.99288, 28.88349], [40.99881, 28.88644]], [[41.019734, 28.81971], [41.06, 28.74877], [41.04447, 28.76268], [41.04043, 28.7706], [41.0578, 28.77666], [41.05292, 28.75341], [41.05471, 28.76214]], [[41.019734, 28.81971], [41.10403, 28.86451], [41.09347, 28.80551], [41.11022, 28.80399], [41.09056, 28.81102], [41.10078, 28.81058], [41.11002, 28.80115], [41.09743, 28.80579], [41.12019, 28.80717], [41.10679, 28.80359], [41.12037, 28.80708], [41.12019, 28.8075], [41.10855, 28.80333]], [[41.019734, 28.81971], [40.98085, 28.79397], [40.98526, 28.79617], [40.97981, 28.79449], [41.00128, 28.77555], [40.99954, 28.78], [40.99657, 28.76948], [40.99498, 28.7912], [40.99323, 28.76885], [40.99393, 28.79101], [40.99824, 28.76625], [40.99404, 28.76789], [40.99699, 28.77676], [40.99368, 28.79092], [40.99928, 28.78542], [40.99659, 28.77578], [40.99812, 28.77808], [40.99623, 28.79155], [40.98804, 28.78032], [40.98884, 28.78093], [40.99639, 28.77529], [40.99363, 28.77587], [40.98919, 28.78033], [40.98986, 28.7864], [40.98833, 28.78191], [40.99277, 28.78173], [40.99501, 28.77398], [40.98679, 28.78414], [40.99197, 28.77848], [40.99107, 28.77691], [40.98703, 28.78417], [41.00214, 28.77796], [40.99197, 28.78394], [40.98655, 28.78256], [40.99093, 28.78424], [40.99096, 28.77703]], [[41.019734, 28.81971], [41.00664, 28.86902], [41.01079, 28.87049], [41.0155, 28.87553], [41.01081, 28.86759], [41.00577, 28.86982], [41.00336, 28.87093], [41.00687, 28.86951], [41.0241, 28.85692], [41.02691, 28.86514], [41.02845, 28.86527], [41.02718, 28.86502], [41.02575, 28.86134], [41.01419, 28.85722], [41.00726, 28.86092], [41.01199, 28.86254], [41.00844, 28.85911], [41.00946, 28.85897], [41.01281, 28.85876], [41.0079, 28.86406], [41.01127, 28.86132], [41.01603, 28.8577], [41.0218, 28.85616], [41.01426, 28.8696], [41.01609, 28.86127], [41.01426, 28.86499], [41.01411, 28.86328], [41.01662, 28.86256], [41.02024, 28.85869], [41.01427, 28.86499], [41.01671, 28.86888], [41.01715, 28.86044], [41.02201, 28.85613], [41.01872, 28.87535], [41.02145, 28.87067], [41.02417, 28.87289], [41.01452, 28.88337], [41.02155, 28.87041], [41.01953, 28.87462], [41.01578, 28.8777]], [[41.019734, 28.81971], [41.03396, 28.79644], [41.03354, 28.77447], [41.0357, 28.78919], [41.03783, 28.7983], [41.03664, 28.78836], [41.03355, 28.80022], [41.03337, 28.80022], [41.03386, 28.80313], [41.03311, 28.79984], [41.04559, 28.78274], [41.04362, 28.78395], [41.03544, 28.79396], [41.0242, 28.80073], [41.02465, 28.79643], [41.02842, 28.7753], [41.02205, 28.7895], [41.02173, 28.78627], [41.02058, 28.79162]], [[41.019734, 28.81971], [41.09623, 28.7909], [41.10239, 28.78701], [41.10226, 28.78703], [41.10033, 28.79022], [41.11218, 28.78581], [41.11199, 28.78498], [41.10704, 28.78471], [41.10802, 28.78975]], [[41.019734, 28.81971], [40.96996, 28.79621], [40.96421, 28.83819], [40.97259, 28.80423], [40.9597, 28.82166], [40.96167, 28.82469], [40.96392, 28.83777], [40.96379, 28.83763]], [[41.019734, 28.81971], [41.02602, 28.82905], [41.0322, 28.82415], [41.03434, 28.8332], [41.03234, 28.82719], [41.02782, 28.8291], [41.03158, 28.82734], [41.0302, 28.82899], [41.03576, 28.82673], [41.04016, 28.82502], [41.03634, 28.83274], [41.03608, 28.83282], [41.0313, 28.82828], [41.03487, 28.82285], [41.0396, 28.82519], [41.02895, 28.82572], [41.03067, 28.83867], [41.02657, 28.83591], [41.02525, 28.83666], [41.0306, 28.83904], [41.03525, 28.84121], [41.03143, 28.844], [41.03075, 28.83783], [41.03983, 28.81678], [41.02142, 28.83977], [41.02245, 28.83912], [41.01965, 28.82418], [41.02255, 28.8392], [41.01837, 28.83188], [41.01786, 28.82593], [41.03779, 28.83134]], [[41.019734, 28.81971], [41.08194, 28.75223], [41.07455, 28.74939], [41.08325, 28.75242], [41.08665, 28.75115], [41.0777, 28.75076], [41.07434, 28.75195], [41.08682, 28.75241], [41.06846, 28.753]]]

    num_runs = 20  # Defined num_runs here
    
    best_final_cost = float('inf')
    best_run_metrics = None
    best_run_clusters = None
    all_runs_initial_metrics = []

    for run_number in range(num_runs):
        initial_metrics, final_metrics, final_clusters = optimize_clustering(cluster_points_list, run_number + 1)
        all_runs_initial_metrics.append(initial_metrics)
        print(f"\n=== RUN {run_number + 1} METRICS ===")
        print(f"INITIAL Clusters: {initial_metrics[0]}, Total Cost: ${initial_metrics[1]:.2f}, Total Time: {initial_metrics[2]:.2f} hrs")
        print(f"FINAL Clusters: {final_metrics[0]}, Total Cost: ${final_metrics[1]:.2f}, Total Time: {final_metrics[2]:.2f} hrs")
        if final_metrics[1] < best_final_cost:
            best_final_cost = final_metrics[1]
            best_run_metrics = final_metrics
            best_run_clusters = final_clusters

    print("\n=== BEST RUN OVERALL ===")
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

        # Second Run Implementation
        if best_run_clusters:
            final_cluster_points = [cluster.data_points.tolist() for cluster in best_run_clusters]

            # Reset parameters for second run (no global declaration needed)
            max_merge_attempts_per_cluster = 30
            max_iterations = 150
            # Note: attempts_left in Cluster class is an instance variable - 
            # you'll need to modify the Cluster class to use this parameter

            # Run second optimization
            initial_metrics2, final_metrics2, final_clusters2 = optimize_clustering(final_cluster_points, num_runs + 1)

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
        print("No valid solution found in any run.")