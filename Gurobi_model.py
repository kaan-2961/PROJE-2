from gurobipy import Model, GRB, quicksum

model = Model("MultiVehicle_VRP")


n_customers = 10   # müşteri sayısı
N = 5              # maksimum araç sayısı
TMAX = 180         # maksimum rota süresi (dakika)

# 0. indeks depo, 1...n_customers müşteri
locations = [(lat, lon), ...]  # koordinatlar

# Mesafeyi/süreyi temsil eden matris
distance_matrix = [[0, 20, ...], [...], ...]

# Girişler
n = len(locations)  # müşteri + depo
K = N

# Karar değişkenleri
x = model.addVars(n, n, K, vtype=GRB.BINARY, name="x")
t = model.addVars(n, K, vtype=GRB.CONTINUOUS, name="t")

# Subtour değişkenleri (MTZ için)
u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

# Amaç: Toplam mesafeyi/süreyi minimize et
model.setObjective(quicksum(distance_matrix[i][j] * x[i, j, k]
                            for i in range(n)
                            for j in range(n) if i != j
                            for k in range(K)), GRB.MINIMIZE)

# Kısıtlar:

# Her müşteri bir kez ziyaret edilmeli
for j in range(1, n):
    model.addConstr(quicksum(x[i, j, k] for i in range(n) if i != j for k in range(K)) == 1)

# Araçlar sadece bir yerden çıkıp bir yere gitsin
for k in range(K):
    model.addConstr(quicksum(x[0, j, k] for j in range(1, n)) <= 1)  # depo çıkışı
    model.addConstr(quicksum(x[i, 0, k] for i in range(1, n)) <= 1)  # depo dönüşü

# Akış kısıtı: gelen ve giden aynı
for k in range(K):
    for h in range(1, n):
        model.addConstr(
            quicksum(x[i, h, k] for i in range(n) if i != h) ==
            quicksum(x[h, j, k] for j in range(n) if j != h)
        )

# Zaman akışı ve zaman sınırı
for k in range(K):
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(t[j, k] >= t[i, k] + distance_matrix[i][j] - 9999 * (1 - x[i, j, k]))

    for i in range(1, n):
        model.addConstr(t[i, k] <= TMAX)

# Subtour elimination (MTZ constraints)
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model.addConstr(u[i] - u[j] + K * x[i, j, 0] <= K - 1)

# Optimize
model.optimize()
