from linear_regression import MyLinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PCA import *


fig, ax = plt.subplots(1, 1, figsize=(10, 10))


data = np.random.rand(100, 5)
X_train = pd.DataFrame(data, columns=[f"f{i}" for i in range(5)])
y_train = 3 * X_train["f0"] + 2 * X_train["f1"] + 5 + np.random.normal(0, 0.1, 100)

model_matrix = MyLinearRegression(
    regularization="l2", weight_calc="matrix", lambda_2=0.1
)
model_matrix.fit(X_train, y_train)

print(f"Matrix R2: {model_matrix.score(X_train, y_train):.4f}")
print(f"Intercept: {model_matrix.intercept_:.2f}")
print(f"Weights: {model_matrix.weights}")


model_sgd = MyLinearRegression(
    regularization="l1l2",
    weight_calc="sgd",
    lambda_1=0.01,
    lambda_2=0.01,
    batch_size=10,
)

model_sgd.fit(X_train, y_train, lr=0.05, max_iter=500)
print(f"SGD R2: {model_sgd.score(X_train, y_train):.4f}")

sk_model = LinearRegression()
sk_model.fit(X_train,y_train)
preds = sk_model.predict(X_train)
print(f"Sklearn R2: {r2_score(y_train, preds):.4f}")
print(f"Sklearn Intercept: {sk_model.intercept_:.2f}")
print(f"Sklearn Coefs: {sk_model.coef_}")

center_data = centered_data(X_train,X_train)
cov_mat = cov_matrix(center_data)
main_compon = get_components(cov_mat)
new_coods = get_new_coords(center_data,main_compon,2)
weights = np.concatenate(([model_matrix.intercept_],model_matrix.coefs_))

w = np.array([
    2.95026876,
    1.94989045,
    0.04966303,
    0.03826767,
    0.02863922
])

lam = 4.99723482

# первая точка (x2=x3=x4=x5=0)
x1 = np.zeros(5)
x1[0] = lam / w[0]

# вторая точка (x2=1, остальные 0)
x2 = np.zeros(5)
x2[1] = 1
x2[0] = (lam - w[1]) / w[0]

# print("point 1:", x1)
# print("point 2:", x2)

# # проверка
# print("w^T x1 =", w @ x1)
# print("w^T x2 =", w @ x2)


new_coords = get_new_coords(np.vstack([x2,x1]),main_compon,2)

print(new_coods[:5])
ax.scatter(new_coods[0],new_coods[1],color = "blue",marker="o")
ax.plot(new_coords[:,[0]],new_coords[:,[1]],color="red")
ax.grid(True)
ax.set_xlabel("Ось X1")
ax.set_ylabel("Ось Y1")
plt.show()