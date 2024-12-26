import cupy as cp


def leray_projection(u_hat: cp.ndarray, k: cp.ndarray, k_k2: cp.ndarray):
    u_hat[:] -= k_k2 * cp.sum(u_hat * k, axis=0)
