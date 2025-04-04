from gurobipy import Model, GRB, quicksum
import numpy as np

locations = [
    (41.0, 29.0),  # depo
    (41.1, 29.1), (41.2, 29.3), (41.3, 29.2),  # müşteriler
    (41.0, 29.4), (41.4, 29.1)
]

n = len(locations)           # tüm noktalar (depo + müşteriler)
K = 3                        # maksimum araç sayısı
TMAX = 180                   # maksimum süre (dk)

# örnek distance matrix (gerçek uygulamada mesafeleri Haversine veya OSM'den al)
distance_matrix = np.random.randint(10, 50, size=(n, n))
np.fill_diagonal(distance_matrix, 0)

#  Model Kurulumu
model = Model("MultiVehicle_VRP_TimeLimited")

x = model.addVars(n, n, K, vtype=GRB.BINARY, name="x")
t = model.addVars(n, K, vtype=GRB.CONTINUOUS, name="t")

# Amaç Fonksiyonu
model.setObjective(quicksum(
    distance_matrix[i][j] * x[i, j, k]
    for i in range(n) for j in range(n) if i != j for k in range(K)
), GRB.MINIMIZE)

#Kısıtlar
# 1. Her müşteri (1..n-1) tam 1 kez ziyaret edilmeli
for j in range(1, n):
    model.addConstr(quicksum(x[i, j, k] for i in range(n) if i != j for k in range(K)) == 1)

# 2. Her araç için depo çıkış ve dönüş kısıtı
for k in range(K):
    model.addConstr(quicksum(x[0, j, k] for j in range(1, n)) <= 1)  # depo çıkışı
    model.addConstr(quicksum(x[i, 0, k] for i in range(1, n)) <= 1)  # depo dönüşü

# 3. Akış dengesi (her müşteri için gelen = giden)
for k in range(K):
    for h in range(1, n):
        model.addConstr(
            quicksum(x[i, h, k] for i in range(n) if i != h) ==
            quicksum(x[h, j, k] for j in range(n) if j != h)
        )

# 4. Zaman akışı ve zaman sınırı
M = 9999  # büyük bir sabit
for k in range(K):
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(t[j, k] >= t[i, k] + distance_matrix[i][j] - M * (1 - x[i, j, k]))

    for i in range(1, n):
        model.addConstr(t[i, k] <= TMAX)

# 5. Subtour elimination (MTZ yöntemi)
u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model.addConstr(u[i] - u[j] + (n-1) * x[i, j, 0] <= n - 2)

# Optimize Et
model.optimize()

# Sonuçları Yazdır
for k in range(K):
    print(f"\nAraç {k+1} rotası:")
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j, k].X > 0.5:
                print(f"{i} -> {j} (süre: {distance_matrix[i][j]} dk)")
