# cv-profet

**Collective Variable Predictions OF Elusive Transitions**

`cv-profet` is a Python toolkit for building collective variables from infRETIS trajectory data and analyzing how well those variables predict rare, reactive transitions. The repository combines two main parts: the CV builder pipeline (`CV_builder/main.py`) and the PPA pipeline (`ppa/analyze.py`), both exposed through one public CLI: `profet`.

## Installation

Install from the repository root with pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Alternatively, use the helper script:

```bash
./install.sh
```

This installs a single public command-line program:

- `profet`

If needed in a custom HPC environment, you can point the launchers to the repository with:

```bash
export CV_PROFET_ROOT=/absolute/path/to/cv-profet
```

## Quick Overview Step by Step

1. `profet build` is the CV matrix builder and should always be the first step.
2. `profet screen` determines the predictive power (`T`) of each CV.
3. `profet diagnose` generates CV diagnostics as a textfile report, overlap and CV distribution plots, and their integrals.
4. optional `profet tmat` builds a T-matrix for varying combinations of `lambda_c` and `lambda_r`.
5. optional `profet optimize` creates linear combinations of CVs and optimizes them based on predictive power `T`.



## Example Commands

Build CVs from a standard run layout:

```bash
profet build --toml infretis.toml
profet build --toml infretis.toml --overwrite
```

Build CVs with explicit input files:

```bash
profet build \
  --toml pp.toml \
  --h5-input simulation.h5 \
  --load-dir load \
  --data infretis_data.txt
```


## How To Add A CV

For the full step-by-step guide, start with [CV_builder/README.md](CV_builder/README.md).

In short:

1. Implement or register your CV in a CV suite.
2. Point `[cv_modules]` in your TOML file to that suite.
3. Pass CV parameters in the same TOML key/value style you already use for order-parameter settings.

Reference implementations:

- [CV_builder/CV_manager/minimal_suite.py](CV_builder/CV_manager/minimal_suite.py) shows how a suite is configured and how TOML parameters are accepted.
- [CV_builder/CV_manager/cvs/cv_distance.py](CV_builder/CV_manager/cvs/cv_distance.py) shows a minimal CV implementation pattern.



## Repository Structure

- `CV_builder/` contains the CV matrix construction pipeline used by `profet build`.
- `ppa/` contains predictive power analysis tools used by `profet screen`, `profet diagnose`, `profet tmat`, and `profet optimize`.
- `run_CV_builder.slurm` shows a typical SLURM submission workflow using `profet` commands.

More detailed documentation:

- [CV_builder/README.md](CV_builder/README.md)
- [ppa/README.md](ppa/README.md)
- [CV_builder/CV_manager/README.md](CV_builder/CV_manager/README.md)
- [CV_builder/topology/README.md](CV_builder/topology/README.md)

## Literature

Primary method papers: 
- [Analyzing Complex Reaction Mechanisms Using Path Sampling](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00642)
- [NaCl Dissociation Explored Through Predictive Power Path Sampling Analysis](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00054)

infRETIS background: 
- [Highly parallelizable path sampling with minimal rejections using asynchronous replica exchange and infinite swaps](https://www.pnas.org/doi/10.1073/pnas.2318731121)
- [Exchanging Replicas with Unequal Cost, Infinitely and Permanently](https://pubs.acs.org/doi/full/10.1021/acs.jpca.2c06004)


Application papers:
- [Local initiation conditions for water autoionization](https://www.pnas.org/doi/10.1073/pnas.1714070115)