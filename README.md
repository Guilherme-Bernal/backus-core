# backus-core

> High-performance numerical linear algebra in Fortran 2018 with Python bindings.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/Guilherme-Bernal/backus-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Guilherme-Bernal/backus-core/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Fortran](https://img.shields.io/badge/Fortran-2018-purple)](https://fortran-lang.org/)

---

Named after **John Backus** — the computer scientist who led the IBM team that created Fortran in 1957, the first widely adopted high-level programming language, and co-creator of the Backus-Naur Form (BNF) notation.

---

## Overview

`backus-core` is a scientific computing library that implements foundational numerical linear algebra algorithms in **Fortran 2018** and exposes them to Python via **f2py**. The goal is to combine Fortran's raw numerical performance with Python's ecosystem for data analysis, testing, and visualization.

| Module | Algorithms | Complexity |
|---|---|---|
| `linalg` | LU decomposition, forward/back substitution | O(n³) -> O(n²) per solve |
| `tridiagonal` | Thomas algorithm | O(n) |
| `norms` | Vector norms (L1, L2, L∞), Frobenius norm | O(n) / O(n²) |

---

## Motivation

Most Python numerical code eventually calls C or Fortran under the hood — NumPy itself is built on LAPACK and BLAS, both written in Fortran. `backus-core` makes this layer explicit: hand-written Fortran 2018 with modern features (explicit interfaces, `intent` declarations, `pure` functions), exposed cleanly to Python.

The project serves two purposes:

1. **Learning** — understanding how foundational algorithms work at the metal level, without abstraction layers hiding the mechanics.
2. **Benchmarking** — measuring real performance differences between a hand-rolled Fortran implementation and NumPy/SciPy equivalents, across matrix sizes from 100x100 to 5000x5000.

---

## Stack

```
Python 3.10+          <- user interface, tests, benchmarks
    |
  f2py                <- auto-generated C bindings (ships with NumPy)
    |
Fortran 2018          <- numerical core (gfortran)
```

---

## Project Structure

```
backus-core/
|
+-- src/                        <- Fortran source
|   +-- linalg/
|   |   +-- lu.f90              <- LU decomposition with partial pivoting
|   |   +-- tridiagonal.f90     <- Thomas algorithm
|   |   +-- norms.f90           <- vector and matrix norms
|   +-- backus.f90              <- root module (re-exports all)
|
+-- python/                     <- Python layer
|   +-- backus/
|   |   +-- __init__.py
|   |   +-- linalg.py           <- Python wrappers with NumPy integration
|   |   +-- _build/             <- compiled .so extensions (generated)
|   +-- tests/
|       +-- test_lu.py
|       +-- test_tridiagonal.py
|       +-- test_norms.py
|
+-- benchmarks/
|   +-- bench_lu.py             <- Fortran vs NumPy vs SciPy
|   +-- bench_tridiagonal.py
|   +-- results/                <- generated plots
|
+-- CMakeLists.txt
+-- pyproject.toml
+-- LICENSE
+-- .github/
    +-- workflows/
        +-- ci.yml
```

---

## Getting Started

### Requirements

- Python 3.10+
- gfortran (GCC 10+)
- NumPy 1.24+ (includes f2py)

### Installation

```bash
git clone git@github.com:Guilherme-Bernal/backus-core.git
cd backus-core

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

pip install numpy scipy pytest matplotlib
```

### Build Fortran extensions

```bash
cd src
python -m numpy.f2py -c linalg/norms.f90 -m backus_norms
```

### Run tests

```bash
pytest python/tests/ -v
```

---

## Modules

### Norms

Vector and matrix norms implemented as Fortran `pure functions`, allowing the compiler to apply aggressive optimizations.

```python
from backus import norms

v = [3.0, 4.0]
print(norms.l2(v))    # 5.0
print(norms.l1(v))    # 7.0
print(norms.linf(v))  # 4.0
```

### LU Decomposition

Factors a dense matrix A into lower (L) and upper (U) triangular matrices with partial pivoting. Solves Ax = b as two triangular substitutions — reducing the cost from O(n³) per solve to O(n²) after the initial factorization.

```python
from backus import linalg
import numpy as np

A = np.array([[2,  1, -1],
              [4,  3,  1],
              [-2, 5,  2]], dtype=np.float64)
b = np.array([1.0, 2.0, 3.0])

x = linalg.lu_solve(A, b)
```

### Tridiagonal Solver

Implements the Thomas algorithm — a specialized O(n) solver for tridiagonal systems. Naturally arises in 1D PDE discretization, cubic spline computation, and heat transfer simulation.

```python
from backus import linalg

lower = [-1.0, -1.0, -1.0]
diag  = [ 2.0,  2.0,  2.0,  2.0]
upper = [-1.0, -1.0, -1.0]
b     = [ 1.0,  0.0,  0.0,  1.0]

x = linalg.tridiagonal_solve(lower, diag, upper, b)
```

---

## Benchmarks

Performance comparison between `backus-core` (Fortran 2018) and NumPy/SciPy across increasing matrix sizes. Results will be updated as each module is completed.

| Matrix size | backus-core | numpy.linalg | scipy.linalg |
|---|---|---|---|
| 100x100 | — | — | — |
| 500x500 | — | — | — |
| 1000x1000 | — | — | — |
| 2000x2000 | — | — | — |
| 5000x5000 | — | — | — |

---

## Roadmap

- [x] Repository structure
- [x] License (GPL v3)
- [ ] `norms.f90` — L1, L2, L∞, Frobenius
- [ ] `lu.f90` — LU decomposition with partial pivoting
- [ ] `tridiagonal.f90` — Thomas algorithm
- [ ] f2py bindings for all modules
- [ ] pytest suite (tolerance 1e-10 vs NumPy reference)
- [ ] Benchmark suite with matplotlib plots
- [ ] GitHub Actions CI
- [ ] pip-installable package (`pyproject.toml`)

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for details.

---

## About

Developed by [Guilherme Savazzi](https://github.com/Guilherme-Bernal).
Computer Engineering student at Facens University, Sao Paulo, Brazil.

> *"Much of my work has come from being lazy."* — John Backus