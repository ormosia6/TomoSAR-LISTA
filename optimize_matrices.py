import numpy as np
import os


def get_matrices(m, n, root_path,matrix_dir):
    """
    Optimize the mutual general coherence of W given phi.

    Avoid recomputing and enforce consistency over multiple runs by saving the results in the matrix_dir.
    """

    d = matrix_dir + str(m) + "_" + str(n) + "/"
    if not os.path.exists(d):
        os.makedirs(d)

    # if not os.path.exists(d + "phi.npy"):
    if not os.path.exists(d + "phi.npy"):
        W = np.random.normal(scale=1 / np.sqrt(m*2), size=(m*2, n*2))
        # W /= np.linalg.norm(W, axis=0).reshape(1, -1)
        # phi = np.random.normal(scale=1 / np.sqrt(m), size=(m, n))
        phi = np.load(os.path.join(root_path,"D.npy"))
        # W = np.random.rand(*D_shape)
        np.save(d + "phi", phi)
        np.save(d + "W", W)
    else:
        phi = np.load(d + "phi.npy")
        W = np.load(d + "W.npy")
        print(f"Using saved phi from {d}.")

    if not os.path.exists(d + "W_frob.npy"):
        W_frob, cohs_frob = opt_frobenius(W, phi, steps=1000)
        np.save(d + "W_frob", W_frob)
    else:
        W_frob = np.load(d + "W_frob.npy")
        print(f"Using saved W from {d}.")

    return phi, W_frob


def proj(W, D):
    m, n = D.shape

    a = np.dot(D.T, W)
    b = np.diag(a)
    c = np.tile(b.reshape(-1, 1), (1, m)).T
    d = W + (1 - c) * D
    return d


def frobenius_norm(W, D):
    return (np.dot(W.T, D) ** 2).sum()


def generalized_coherence(W, D):
    gram = np.dot(W.T, D) - np.eye(W.shape[1])
    return np.max(np.abs(gram))


def opt_frobenius(W, D, steps=1000):
    eta = 0.1
    coherences = []

    for it in range(steps):
        W = proj(W - eta * D.dot(D.T.dot(W)), D)

        diag = np.diag(np.dot(W.T, D))

        W /= diag
        val_ = frobenius_norm(W, D)
        eta *= 0.99
        coherences.append(generalized_coherence(W, D))
        if it % 100 == 0:
            print(
                "Iteration", it, "F-Norm", val_, "Generalized Coherence:", coherences[-1],
            )

    return W, coherences
