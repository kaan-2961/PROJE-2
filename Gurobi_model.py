from gurobipy import Model, GRB, quicksum
import numpy as np

#  Parametreler
speed_km_per_hr = 35
service_time_hr = 0.05       # 3 dk
tmax = 3                     # saat
hiring_cost_per_cluster = 50
distance_cost_per_km = 2

#  Lokasyonlar (0 = depo, 1-... = müşteriler)
locations = [
    (41.0, 29.0),  # depo
    (41.1, 29.1), (41.2, 29.3), (41.3, 29.2),
    (41.0, 29.4), (41.4, 29.1), (41.2, 29.0),
    (41.3, 29.3), (41.1, 29.4), (41.0, 29.2),
    (41.2, 29.2)
]

n = len(locations)  # toplam nokta sayısı (depo + müşteriler)
K = 3               # araç sayısı

# Mesafe ve süre matrisi oluştur
def haversine_km(coord1, coord2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

distance_matrix_km = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            distance_matrix_km[i][j] = haversine_km(locations[i], locations[j])

distance_matrix_hr = distance_matrix_km / speed_km_per_hr  # süre (saat)

#  Gurobi Modeli
model = Model("MultiVehicle_TSP_TimeLimited_Cost")

# Karar değişkenleri
x = model.addVars(n, n, K, vtype=GRB.BINARY, name="x")
t = model.addVars(n, K, vtype=GRB.CONTINUOUS, name="t")

# Amaç fonksiyonu (mesafe maliyeti + araç kullanımı maliyeti)
model.setObjective(
    quicksum(distance_cost_per_km * distance_matrix_km[i][j] * x[i, j, k]
             for i in range(n) for j in range(n) if i != j for k in range(K)) +
    quicksum(hiring_cost_per_cluster * quicksum(x[0, j, k] for j in range(1, n)) for k in range(K)),
    GRB.MINIMIZE
)

#  Kısıtlar

# 1. Her müşteri 1 kez ziyaret edilmeli
for j in range(1, n):
    model.addConstr(quicksum(x[i, j, k] for i in range(n) if i != j for k in range(K)) == 1)

# 2. Her araç en fazla 1 kez çıkış ve 1 kez dönüş yapabilir
for k in range(K):
    model.addConstr(quicksum(x[0, j, k] for j in range(1, n)) <= 1)
    model.addConstr(quicksum(x[i, 0, k] for i in range(1, n)) <= 1)

# 3. Akış dengesi
for k in range(K):
    for h in range(1, n):
        model.addConstr(
            quicksum(x[i, h, k] for i in range(n) if i != h) ==
            quicksum(x[h, j, k] for j in range(n) if j != h)
        )

# 4. Zaman akışı ve tmax kısıtı
M = 9999
for k in range(K):
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(
                    t[j, k] >= t[i, k] + distance_matrix_hr[i][j] + service_time_hr - M * (1 - x[i, j, k])
                )
    for i in range(1, n):
        model.addConstr(t[i, k] <= tmax)

# 5. Subtour elimination (MTZ yöntemi)
u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model.addConstr(u[i] - u[j] + (n - 1) * x[i, j, 0] <= n - 2)

model.optimize()

#  Çıktılar
if model.SolCount > 0:
    total_cost = model.ObjVal
    print(f"\n✅ Çözüm Bulundu! Toplam Maliyet: {total_cost:.2f} TL\n")
    for k in range(K):
        print(f"🛻 Araç {k+1} rotası:")
        for i in range(n):
            for j in range(n):
                if i != j and x[i, j, k].X > 0.5:
                    print(f"{i} → {j} | Mesafe: {distance_matrix_km[i][j]:.2f} km | Süre: {distance_matrix_hr[i][j]*60:.1f} dk + servis")
else:
    print("⚠️ Model için çözüm bulunamadı (infeasible).")
