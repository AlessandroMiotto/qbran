"""Unitary gauge transformations for superconducting circuits.

Provides graph-theoretic gauge rotation algorithms that transform complex Hermitian
Hamiltonians into strictly real symmetric (float64) representations for arbitrary
topologies (trees, linear chains, star networks, multi-transmon, multi-resonator).
"""

from typing import Tuple, Union, Optional
import numpy as np
import scipy.sparse as sp


def is_strictly_real(H: Union[np.ndarray, sp.csr_matrix], tol: float = 1e-12) -> bool:
    """Check if matrix H has zero imaginary part within numerical tolerance."""
    if not np.iscomplexobj(H):
        return True
    if sp.issparse(H):
        if H.data.size == 0:
            return True
        return bool(np.max(np.abs(H.data.imag)) < tol)
    return bool(np.max(np.abs(H.imag)) < tol)


def make_real_gauge(
    H: Union[np.ndarray, sp.csr_matrix],
    sparse: Optional[bool] = None,
    tol: float = 1e-12,
) -> Tuple[Union[np.ndarray, sp.csr_matrix], np.ndarray, bool]:
    """Transform a Hermitian matrix H into a strictly real symmetric matrix (float64).

    Finds a diagonal unitary matrix D = diag(exp(i * theta)) such that:
        H_real = (D^dagger @ H @ D).real
    is strictly real symmetric with machine precision.

    Parameters
    ----------
    H : np.ndarray or sp.csr_matrix
        Hermitian Hamiltonian matrix.
    sparse : bool, optional
        Whether to return a sparse CSR matrix or dense NumPy array.
        Defaults to matching the input matrix format.
    tol : float, optional
        Numerical tolerance for detecting zero imaginary parts and cycle phases.

    Returns
    -------
    H_real : np.ndarray or sp.csr_matrix
        Strictly real symmetric matrix (dtype=np.float64).
    D : np.ndarray
        1D array of diagonal unitary phase factors exp(i * theta).
    success : bool
        True if gauge transformation was successful; False if obstructed by non-zero
        geometric/loop flux (in which case the input H is returned as-is).
    """
    dim = H.shape[0]
    is_sp = sp.issparse(H) if sparse is None else bool(sparse)

    # Already real or negligibly imaginary
    if not np.iscomplexobj(H):
        H_out = H if is_sp == sp.issparse(H) else (H.tocsr() if is_sp else H.toarray())
        return H_out.astype(np.float64), np.ones(dim, dtype=np.complex128), True

    if is_strictly_real(H, tol=tol):
        H_real = H.real.astype(np.float64)
        if is_sp and not sp.issparse(H_real):
            H_real = sp.csr_matrix(H_real)
        elif not is_sp and sp.issparse(H_real):
            H_real = H_real.toarray()
        return H_real, np.ones(dim, dtype=np.complex128), True

    # Graph Breadth-First Search (BFS) for spanning tree gauge phases
    csr = H.tocsr() if sp.issparse(H) else sp.csr_matrix(H)
    visited = np.zeros(dim, dtype=bool)
    theta = np.zeros(dim, dtype=np.float64)

    for start_node in range(dim):
        if visited[start_node]:
            continue
        visited[start_node] = True
        queue = [start_node]

        while queue:
            u = queue.pop(0)
            row = csr.getrow(u)
            for v, val in zip(row.indices, row.data):
                if u == v or abs(val) < tol:
                    continue

                # Condition: arg(H_uv) + theta_v - theta_u = 0 (mod pi)
                # Therefore: theta_v = theta_u - arg(H_uv) (mod pi)
                phase_target = theta[u] - np.angle(val)

                if not visited[v]:
                    visited[v] = True
                    theta[v] = phase_target
                    queue.append(v)
                else:
                    # Check cycle consistency (loop phase must be integer multiple of pi)
                    diff = (phase_target - theta[v]) / np.pi
                    if abs(diff - round(diff)) > 1e-4:
                        # Non-zero gauge-invariant loop flux obstructs real representation
                        H_out = H if is_sp == sp.issparse(H) else (H.tocsr() if is_sp else H.toarray())
                        return H_out, np.ones(dim, dtype=np.complex128), False

    D = np.exp(1j * theta)

    if sp.issparse(H):
        D_diag = sp.diags(D, 0, format="csr", dtype=np.complex128)
        D_conj = sp.diags(D.conj(), 0, format="csr", dtype=np.complex128)
        H_trans = D_conj @ H @ D_diag
        H_real = H_trans.real.astype(np.float64)
        if not is_sp:
            H_real = H_real.toarray()
        return H_real, D, True
    else:
        H_trans = D.conj()[:, None] * H * D[None, :]
        H_real = H_trans.real.astype(np.float64)
        if is_sp:
            H_real = sp.csr_matrix(H_real)
        return H_real, D, True


def apply_gauge_to_evecs(evecs: np.ndarray, D: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Transform eigenvectors between original and real-gauge representations.

    If H_real = D^dagger H_orig D, then:
        v_real = D^dagger v_orig  ==>  v_orig = D v_real

    Parameters
    ----------
    evecs : np.ndarray
        Array of eigenvectors (columns are eigenvectors).
    D : np.ndarray
        1D diagonal gauge factors.
    inverse : bool, optional
        If True, transforms v_orig -> v_real. Defaults to False (v_real -> v_orig).
    """
    if inverse:
        factor = D.conj()
    else:
        factor = D
    return factor[:, None] * evecs
