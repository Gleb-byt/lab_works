import pandas as pd
import numpy as np
import math


class MyLinearRegression:
    """
    Parameters
    ----------
    regularization : {None, 'l1', 'l2', 'l1l2'}, default=None
        Какую регуляризацию добавить к модели. Если значение `None`, то без регуляризации.

    weight_calc : {'matrix', 'gd', 'sgd'}, default='matrix'
        Каким образом вычислять вектор весов: матрично ('matrix'), градиентным спуском ('gd') или стохастическим градиентным спуском ('sgd'). При этом, при 'l1' или 'l1l2' нельзя использовать параметр 'matrix'.

    Attributes
    ----------
    coefs_ : Вектор коэффициентов размера (p, 1), где p — количество признаков.
    intercept_ : Значение коэффициента, отвечающего за смещение
    """

    def __init__(
        self,
        regularization=None,
        weight_calc="matrix",
        lambda_1=None,
        lambda_2=None,
        batch_size=20,
    ):
        if regularization not in [None, "l1", "l2", "l1l2"]:
            raise TypeError(
                f"Параметр regularization не может принимать значение '{regularization}'"
            )
        if weight_calc not in ["matrix", "gd", "sgd"]:
            raise TypeError(
                f"Параметр weight_calc не может принимать значение '{weight_calc}'"
            )
        if regularization in ["l1", "l1l2"] and lambda_1 is None:
            raise TypeError(f"Значение коэффициента регулризации l1 не задано")
        if regularization in ["l2", "l1l2"] and lambda_2 is None:
            raise TypeError(f"Значение коэффициента регулризации l2 не задано")
        if regularization in ["l1", "l1l2"] and weight_calc == "matrix":
            raise ValueError(
                f"Нельзя использовать и регулиризацию (l1,l1l2) и параметр matrix"
            )

        self.regularization = regularization
        self.weight_calc = weight_calc
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.batch_size = batch_size

        self.coefs_ = None
        self.intercept_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        lr: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        X_arr = np.array(X)
        y_arr = np.array(y)
        if len(y_arr.shape) > 1:
            y_arr = y_arr.flatten()
        ones_col = np.ones((X_arr.shape[0], 1))
        X_full = np.hstack([ones_col, X_arr])
        n_samples, n_features = X_full.shape
        if self.weight_calc == "matrix":
            self._fit_matrix(X_full, y_arr, n_features)
        else:
            self._fit_gradient(
                X_full, y_arr, n_samples, n_features, lr, max_iter, random_state
            )
        self.coefs_ = self.weights[1:]
        self.intercept_ = self.weights[0]

    def _fit_matrix(self, X, y, n_features):
        if self.regularization == "l2":
            i = np.eye(n_features)
            i[0, 0] = 0
            lhs = X.T @ X + self.lambda_2 * i
            rhs = X.T @ y
            self.weights = np.linalg.solve(lhs, rhs)
        else:
            lhs = X.T @ X
            rhs = X.T @ y
            self.weights = np.linalg.solve(lhs, rhs)

    def _fit_gradient(self, X, y, n_samples, n_features, lr, max_iter, random_state):
        np.random.seed(random_state)
        self.weights = np.random.randn(n_features) * 0.01
        for _ in range(max_iter):
            if self.weight_calc == "gd":
                grad = self._calc_gradient(X, y, self.weights, n_samples)
                self.weights -= lr * grad
            elif self.weight_calc == "sgd":
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                    current_batch_size = X_batch.shape[0]
                    grad = self._calc_gradient(
                        X_batch, y_batch, self.weights, current_batch_size
                    )
                    self.weights -= lr * grad

    def _calc_gradient(self, X, y, w, n):
        predictions = X @ w
        error = predictions - y
        grad = (2 / n) * (X.T @ error)
        w_reg = w.copy()
        w_reg[0] = 0
        if self.regularization in ["l1", "l1l2"]:
            grad += self.lambda_1 * np.sign(w_reg)

        if self.regularization in ["l2", "l1l2"]:
            grad += 2 * self.lambda_2 * w_reg

        return grad

    def predict(self, X: np.array, ss=True):
        X_arr = np.array(X)
        ones_col = np.ones((X_arr.shape[0], 1))
        X_full = np.hstack([ones_col, X_arr])
        full_weights = np.concatenate(([self.intercept_], self.coefs_.flatten()))
        return X_full @ full_weights

    def score(self, X: np.array, y: np.array):
        y_pred = self.predict(X)
        y_true = np.array(y)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    arr_1 = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    arr_2 = np.array(range(1, 5))
    # print(x := arr_1.shape, len(x))
    # print(x := arr_2.shape, len(x))
    # print(arr_1.flatten())
    arr_3 = np.ones((4, 1))
    # print(np.hstack([arr_1, arr_3]))
    # print(np.eye(4))
    np.random.seed(42)
    print(np.random.randn(4) * 0.01)
    indices = np.random.permutation(4)
    # print(arr_1[indices])
    # print(arr_1[2:, [0]])
    # print(arr_1.flatten().reshape(-1, 1))
    # print(arr_1.reshape(-1, 1))
    # print(arr_1.flatten().reshape(-1, 1).shape)
    print(np.random.normal(0, 0.1, 10))
    print(np.concatenate(([1], [2, 3, 4, 5])))
    print(np.mean(arr_1))
    print(np.sum(arr_1) / math.prod(arr_1.shape))
    print(np.prod(arr_1))
    print(math.prod(arr_1))
