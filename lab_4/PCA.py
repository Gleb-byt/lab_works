import numpy as np


def centered_data(test_X: np.array, train_X: np.array) -> np.array:
    mu_train = np.mean(train_X, axis=0)
    centered_test_X = test_X - mu_train
    return centered_test_X


def cov_matrix(mat: np.array) -> np.array:
    return (mat.T @ mat) / (mat.shape[0] - 1)


def power_iteration(A: np.array, num_iter: int = 1000, tol: float = 1e-10):
    n = A.shape[0]
    v = np.ones(n)
    for _ in range(num_iter):
        Av = A @ v
        a = (v @ Av) / (v @ v)
        norm_av = np.sqrt(Av @ Av)
        if norm_av == 0:
            return 0, np.zeros(n)
        v_next = Av / np.sqrt(Av @ Av)
        if all(abs(v[i] - v_next[i]) < tol for i in range(n)):
            break
        v = v_next
    return a, v


def deflate(A: np.array, a: int | float, v: np.array):
    return A - a * (v.reshape(-1, 1) @ v.reshape(1,-1))


def get_components(cov_matrix):
    A = np.array(cov_matrix, copy=True)
    n = A.shape[0]
    eigenvalues = list()
    eigenvectors = list()
    for _ in range(n):
        a, v = power_iteration(A)
        eigenvalues.append(a)
        eigenvectors.append(v)
        A = deflate(A, a, v)
    eig_pairs = list(zip(eigenvalues, eigenvectors))
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    sorted_vecs = [vec for _, vec in eig_pairs]
    components_matrix = np.array(sorted_vecs).T
    return components_matrix


def get_new_coords(centered_data, components, n_components):
    W = components[:, :n_components]
    Z = centered_data @ W
    return Z


if __name__ == "__main__":
    arr_1 = np.array([[1, 2], [2, 3], [3, 4]])
    print(arr_1 - np.mean(arr_1, axis=0))
    print(np.ones(4))
    print(arr_1.shape)
    print(arr_1 @ np.array([1, 1]))
    arr_2 = np.ones(5)
    print(arr_2.reshape(-1, 1))
    print(arr_2.reshape(-1, 1).T)
