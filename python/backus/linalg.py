import numpy as np

try:
    from backus._build import backus_norms as _backus
    _norms = _backus.linalg_norms
    _FORTRAN_AVAILABLE = True
except ImportError:
    _FORTRAN_AVAILABLE = False


def _require_fortran():
    if not _FORTRAN_AVAILABLE:
        raise RuntimeError(
            "Fortran extensions not compiled. "
            "Run: python -m numpy.f2py -c src/linalg/norms.f90 -m backus_norms"
        )


def _validate_vector(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got shape {v.shape}. "
            "Use norm_frobenius() for matrices."
        )
    if v.size == 0:
        raise ValueError("Array must not be empty.")
    return np.asfortranarray(v)


def _validate_matrix(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got shape {A.shape}. "
            "Use norm_l1/l2/linf() for vectors."
        )
    if A.size == 0:
        raise ValueError("Matrix must not be empty.")
    return np.asfortranarray(A)


def norm_l1(v):
    _require_fortran()
    v = _validate_vector(v)
    return float(_norms.norm_l1(v, v.size))


def norm_l2(v):
    _require_fortran()
    v = _validate_vector(v)
    return float(_norms.norm_l2(v, v.size))


def norm_linf(v):
    _require_fortran()
    v = _validate_vector(v)
    return float(_norms.norm_linf(v, v.size))


def norm_frobenius(A):
    _require_fortran()
    A = _validate_matrix(A)
    m, n = A.shape
    return float(_norms.norm_frobenius(A, m, n))