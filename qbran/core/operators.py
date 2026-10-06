"""Core quantum operators for superconducting circuits.

All operators support both NumPy (float64/complex128) and SciPy sparse representations.
"""

from typing import Union
import numpy as np
import scipy.sparse as sp


def charge_operator(ncut: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Charge number operator n = diag(-ncut, ..., ncut) in Cooper pair charge basis.
    
    Dimension is 2*ncut + 1. Returns strict real diagonal matrix.
    """
    if ncut < 1:
        raise ValueError(f"ncut must be a positive integer, got {ncut}")
    diag = np.arange(-ncut, ncut + 1, dtype=np.float64)
    if sparse:
        return sp.diags(diag, 0, format='csr', dtype=np.float64)
    return np.diag(diag)


def cos_phi_operator(ncut: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Cosine of phase operator cos(phi) = 0.5 * sum_n (|n><n+1| + |n+1><n|).
    
    Dimension is 2*ncut + 1. Returns strict real symmetric tridiagonal matrix.
    """
    if ncut < 1:
        raise ValueError(f"ncut must be a positive integer, got {ncut}")
    dim = 2 * ncut + 1
    off_diag = np.full(dim - 1, 0.5, dtype=np.float64)
    if sparse:
        return sp.diags([off_diag, off_diag], [-1, 1], shape=(dim, dim), format='csr', dtype=np.float64)
    return np.diag(off_diag, -1) + np.diag(off_diag, 1)


def sin_phi_operator(ncut: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Sine of phase operator sin(phi) = -0.5j * sum_n (|n><n+1| - |n+1><n|).
    
    Dimension is 2*ncut + 1. Returns antisymmetric complex tridiagonal matrix.
    """
    if ncut < 1:
        raise ValueError(f"ncut must be a positive integer, got {ncut}")
    dim = 2 * ncut + 1
    off_diag_upper = np.full(dim - 1, -0.5j, dtype=np.complex128)
    off_diag_lower = np.full(dim - 1, 0.5j, dtype=np.complex128)
    if sparse:
        return sp.diags([off_diag_lower, off_diag_upper], [-1, 1], shape=(dim, dim), format='csr', dtype=np.complex128)
    return np.diag(off_diag_lower, -1) + np.diag(off_diag_upper, 1)


def annihilation_operator(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Bosonic annihilation operator a = sum_n sqrt(n) |n-1><n| in Fock basis.
    
    Dimension is dim. Returns strict real super-diagonal matrix.
    """
    if dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim}")
    if dim == 1:
        return sp.csr_matrix((1, 1), dtype=np.float64) if sparse else np.zeros((1, 1), dtype=np.float64)
    super_diag = np.sqrt(np.arange(1, dim, dtype=np.float64))
    if sparse:
        return sp.diags([super_diag], [1], shape=(dim, dim), format='csr', dtype=np.float64)
    return np.diag(super_diag, 1)


def creation_operator(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Bosonic creation operator a^dagger = sum_n sqrt(n+1) |n+1><n| in Fock basis.
    
    Dimension is dim. Returns strict real sub-diagonal matrix.
    """
    if dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim}")
    if dim == 1:
        return sp.csr_matrix((1, 1), dtype=np.float64) if sparse else np.zeros((1, 1), dtype=np.float64)
    sub_diag = np.sqrt(np.arange(1, dim, dtype=np.float64))
    if sparse:
        return sp.diags([sub_diag], [-1], shape=(dim, dim), format='csr', dtype=np.float64)
    return np.diag(sub_diag, -1)


def number_operator(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Bosonic number operator n = a^dagger * a = diag(0, 1, 2, ..., dim-1) in Fock basis.
    
    Dimension is dim. Returns strict real diagonal matrix.
    """
    if dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim}")
    diag = np.arange(dim, dtype=np.float64)
    if sparse:
        return sp.diags(diag, 0, format='csr', dtype=np.float64)
    return np.diag(diag)


def quadrature_x(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Quadrature operator x = (a + a^dagger) / sqrt(2) in Fock basis.
    
    Dimension is dim. Returns strict real symmetric tridiagonal matrix.
    """
    a = annihilation_operator(dim, sparse=sparse)
    a_dag = creation_operator(dim, sparse=sparse)
    return (a + a_dag) / np.sqrt(2.0)


def quadrature_p(dim: int, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
    """Quadrature operator p = -i * (a - a^dagger) / sqrt(2) in Fock basis.

    Dimension is dim. Returns Hermitian matrix with purely imaginary off-diagonal elements.
    """
    a = annihilation_operator(dim, sparse=sparse)
    a_dag = creation_operator(dim, sparse=sparse)
    return -1j * (a - a_dag) / np.sqrt(2.0)