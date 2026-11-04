# parallel-sindy

[![arXiv](https://img.shields.io/badge/arXiv-2611.05215-b31b1b.svg)](https://arxiv.org/abs/2611.05215)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Parallel Sparse Identification of Nonlinear Dynamics (SINDy) in Python**: task- and
data-parallel SINDy with Numba `prange`, OpenMP (PyOMP), Python threads and
`multiprocessing`, applied to the Lorenz and Rössler systems.

This repository contains the code, data and scripts for the paper

> P. Guillen-Rondon, *Parallel Sparse Identification of Nonlinear Dynamics in Python:
> Task- and Data-Parallel SINDy with Numba, OpenMP, Threads and Multiprocessing for the
> Lorenz and Rössler Systems*, [arXiv:2611.05215](https://arxiv.org/abs/2611.05215) (2026).

SINDy was introduced by Brunton, Proctor & Kutz,
[PNAS 113, 3932 (2016), arXiv:1509.03580](https://arxiv.org/abs/1509.03580).
Results are validated against [PySINDy](https://github.com/dynamicslab/pysindy).

## Highlights

* One set of loop-based SINDy kernels (RK4 integrator, finite differences, polynomial
  library, normal equations, STLSQ) compiled with Numba and run by **four parallel back
  ends**: Numba `prange`, PyOMP (`with openmp("parallel for")`), GIL-free Python threads,
  and `multiprocessing` with shared memory.
* Two parallel decompositions:
  * **ensemble** (task parallelism): many independent SINDy fits, e.g. 960 trajectories;
  * **library** (data parallelism): one long trajectory (2×10⁷ samples, 56 candidate
    terms), with per-worker partial sums of ΘᵀΘ and ΘᵀẊ that are reduced.
* A cache-blocked normal-equation formulation (Θ is never stored), with an automatic
  QR (TSQR) fallback for ill-conditioned data.
* Identified models match PySINDy in every case; on a 24-core node the serial code is
  4.3–5.6× faster than PySINDy, parallel speed-ups reach 14.8–20.3, and the overall time
  to solution drops by up to 117×.

## Installation

```bash
git clone https://github.com/guillenrondonp/parallel-sindy.git
cd parallel-sindy
```


Python ≥ 3.10 is required (tested with Python 3.12). Installing the compiled
packages from conda-forge is the most reliable route, also on clusters with an older Linux:

```bash
conda create -n sindy -c conda-forge python=3.12 numpy numba matplotlib scipy scikit-learn
conda activate sindy
pip install pysindy                                            # reference implementation
conda install -c python-for-hpc -c conda-forge pyomp           # optional: OpenMP back end
```

PyOMP supports Linux x86-64/ARM64 and macOS ARM64; without it the other back ends still run.
Check the installation:

```bash
python -c "import numpy, numba, pysindy; from multiprocessing import shared_memory; print('ok')"
python -c "from numba.openmp import njit; print('PyOMP ok')"
```

## Quick start

```bash
# a two-minute check on any machine
python benchmark.py --quick --cores 1 2 4

# one case by hand, printing the identified equations
python run_backend.py --system lorenz --workload library --backend numba --cores 8 --show
```

`--show` prints, for example,

```
x' = -9.999 x +9.999 y
y' = +27.992 x -0.998 y -1.000 x z
z' = -2.666 z +1.000 x y
```

## Reproducing the paper

The paper's benchmark ran on one exclusive compute node of the UHD HPC cluster
(2 × Intel Xeon E5-2650 v4, 24 cores, 48 hardware threads):

```bash
python benchmark.py --cores 1 2 4 8 12 16 24 48 --M 960 --N 20000000 --out results_node
# or, under Slurm:
sbatch hpc/run_node.sbatch
```

This takes about 3 h on that node. Every line reports the number of correctly identified
models (`correct models=960/960`). Then regenerate the figures and the manuscript:

```bash
python paper/verify_accuracy.py                         # accuracy check + identified coefficients
python paper/make_figures.py results_node/results.json  # figures -> paper/figures/
cd paper && npm install docx && node build_paper.js ../results_node/results.json
```

The data used in the paper are in [`results_uhd/`](results_uhd/): `results.json` (all
timings), `summary.md` (tables) and `sindy_node_1298.out` (log, including the hardware).

## Repository layout

| path | content |
|---|---|
| `sindy_core.py` | SINDy kernels written once and compiled by any `jit` through `build_kernels(jit)` |
| `backends_numba.py` | serial, `prange`, threads (`nogil`) and `multiprocessing` (shared memory) back ends |
| `backends_pyomp.py` | OpenMP back end with PyOMP |
| `run_backend.py` | runs one (system, workload, back end, cores) case in a fresh process and prints JSON |
| `benchmark.py` | strong-scaling sweep → `summary.md`, `results.json`, speed-up / efficiency / time plots |
| `paper/` | scripts that produce the figures, the accuracy check and the manuscript |
| `results_uhd/` | raw benchmark results of the paper |
| `hpc/` | Slurm job scripts; `run_node.sbatch` is the one used for the paper |
| `run_mpi.py`, `hpc/2*_mpi_*.sbatch` | **experimental** multi-node MPI back end (future work, see `hpc/README_HPC.md`) |

## How the back ends work

```python
@njit(parallel=True)                       # Numba prange: one SINDy fit per iteration
def ensemble_prange(system, p, x0s, dt, n, expo, thr):
    out = np.empty((x0s.shape[0], expo.shape[0], 3))
    for t in prange(x0s.shape[0]):
        out[t] = sindy_one(system, p, x0s[t], dt, n, expo, thr)
    return out

@njit                                      # PyOMP == #pragma omp parallel for
def ensemble_omp(system, p, x0s, dt, n, expo, thr):
    ...
    with openmp("parallel for schedule(static)"):
        for t in range(M):
            out[t] = sindy_one(system, p, x0s[t], dt, n, expo, thr)
```

The thread back end calls the same kernels compiled with `nogil=True` from a
`ThreadPoolExecutor`; the process back end uses a `multiprocessing.Pool` whose workers
attach to the trajectory through `multiprocessing.shared_memory`.

## Notes on measurements

* Speed-up is relative to the compiled serial code, `S(p) = T_serial / T(p)`; the gain
  with respect to PySINDy is reported separately.
* Each case runs in a fresh interpreter; thread counts are fixed before Numba is imported
  and BLAS is pinned to one thread.
* Times are the best of three runs after a warm-up (JIT compilation, pool start-up).
* Use physical cores: on the test node, hyper-threads add only 4–37 %.

## Citation

If you use this code, please cite the paper and the original SINDy article
(Brunton, Proctor & Kutz, PNAS 2016):

```bibtex
@article{guillenrondon2026parallelsindy,
  title         = {Parallel Sparse Identification of Nonlinear Dynamics in Python:
                   Task- and Data-Parallel {SINDy} with {Numba}, {OpenMP}, Threads and
                   Multiprocessing for the {Lorenz} and {R\"ossler} Systems},
  author        = {Guillen-Rondon, Pablo},
  journal       = {arXiv preprint arXiv:2611.05215},
  year          = {2026},
  eprint        = {2611.05215},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2611.05215},
  url           = {https://arxiv.org/abs/2611.05215}
}
```

GitHub's "Cite this repository" button (from [`CITATION.cff`](CITATION.cff)) gives the same reference.

## Acknowledgments

The author acknowledges the High-Performance Computing Center of the College of Sciences
and Technology at the University of Houston–Downtown, operated by UHD Information
Technology, for the computational resources used in this work.

## License

[MIT](LICENSE) © 2026 Pablo Guillen-Rondon
