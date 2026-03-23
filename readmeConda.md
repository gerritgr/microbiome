# Conda Setup for macOS Apple Silicon and Running `gai_cal.py`

This guide is optimized for a native Apple Silicon Mac such as an M3 using Conda on `osx-arm64`.

## Important constraint

Do **not** use the `osx-64` / Rosetta approach in your current full Anaconda setup.

Why:

- your Conda base installation includes the `_anaconda_depends` metapackage
- cross-platform solving with `osx-64` can fail with errors like:

```text
InvalidSpec: The package "_anaconda_depends ..." is not available for the specified platform
```

So for your machine, the correct approach is: stay native on `osx-arm64`.

## Short recommendation

For [gai_cal.py](/Users/gerritprivat/Documents/devel/curahack-2026-challenge-5/gai_cal.py) on an M3 Mac:

1. create a fresh native `osx-arm64` environment
2. install compiled scientific packages with `conda-forge`
3. install the remaining packages with `pip`
4. force wheel-only installs where possible so `pip` does not try to build old packages from source

## 1. Create a fresh environment

```bash
conda remove -n gutenv --all -y
conda create -n gutenv python=3.11 pip -y
conda activate gutenv
```

## 2. Install the compiled scientific stack with Conda Forge

Use Conda for the packages that most often break on macOS when built from source:

```bash
conda install -c conda-forge --strict-channel-priority \
  "numpy<2" \
  pandas \
  scipy \
  matplotlib \
  seaborn \
  scikit-learn \
  statsmodels \
  openpyxl \
  h5py \
  numba \
  networkx \
  xgboost \
  lightgbm \
  llvm-openmp \
  joblib \
  requests \
  tqdm \
  jinja2 \
  plotly \
  psutil \
  -y
```

Notes:

- `llvm-openmp` is the correct OpenMP runtime package on Apple Silicon
- `lightgbm` should be installed from Conda on macOS, not from `pip`
- `numpy`, `pandas`, `scipy`, and `scikit-learn` should also come from Conda to avoid binary mismatch issues

## 3. Install the remaining packages with pip

Install the non-core packages with wheel-only / binary-preferred settings:

```bash
pip install --prefer-binary --only-binary=:all: \
  catboost \
  biom-format \
  scikit-bio \
  redbiom
```

Then install PyCaret:

```bash
pip install --prefer-binary pycaret==3.3.2
```

Why `pycaret==3.3.2`:

- it is the latest stable PyCaret 3 release listed on PyPI
- the repo README says to use the latest PyCaret rather than the paper’s original exact environment

PyPI source:

- [PyCaret on PyPI](https://pypi.org/project/pycaret/)

## 4. If `pycaret` still tries to build old dependencies

If you still get errors such as:

- `Failed to build 'scikit-learn'`
- NumPy metadata-generation failures
- `numpy.distutils` / `CCompiler` errors

then `pip` is still trying to build an old dependency from source.

In that case, do this:

```bash
pip install --prefer-binary --no-build-isolation pycaret==3.3.2
```

If that still fails, the practical conclusion is:

- `gai_cal.py` is too dependent on a fragile PyCaret stack for your native M3 environment
- the better engineering choice is to replace the PyCaret part with a direct scikit-learn / CatBoost / LightGBM pipeline instead of forcing the original script

## 5. Packages you should not install by default in this environment

Do **not** install `shap` in this environment unless you specifically need it later.

Reason:

- recent `shap` versions can pull in `numpy>=2`
- that can conflict with the PyCaret-compatible stack used here

If you need model-interpretation tooling later, use a separate environment.

## 6. Troubleshooting

### `libomp.dylib` / OpenMP error

If you see:

```text
Library not loaded: @rpath/libomp.dylib
```

run:

```bash
conda install -c conda-forge llvm-openmp lightgbm -y
```

### NumPy / Pandas binary mismatch

If you see:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

your environment mixed incompatible compiled binaries.

Fix:

```bash
conda remove -n gutenv --all -y
```

Then recreate the environment and install `numpy`, `pandas`, `scipy`, and `scikit-learn` from Conda Forge first.

### `CCompiler` / `numpy.distutils` / `metadata-generation-failed`

If you see errors like:

```text
NameError: name 'CCompiler' is not defined
error: metadata-generation-failed
ERROR: Failed to build 'scikit-learn' when installing build dependencies
```

that means `pip` is trying to compile an old package from source on Apple Silicon.

That is exactly what this guide is trying to avoid.

Actions:

1. rebuild the environment
2. install the compiled scientific stack from Conda Forge first
3. use `pip install --prefer-binary`
4. if needed, use `pip install --no-build-isolation pycaret==3.3.2`

## 7. Repository layout

From the root folder, the relevant files are:

```text
.
├── gai_cal.py
├── prepare_data.py
├── data/
│   └── datasets/
│       └── processed/
│           ├── AGP/
│           └── GGMP/
└── ...
```

`gai_cal.py` expects plain TSV files on disk:

- `meta.tsv`
- `otu.tsv`

In your current setup, the processed files are already extracted and available at:

```text
data/datasets/processed/AGP/meta.tsv
data/datasets/processed/AGP/otu.tsv
data/datasets/processed/GGMP/meta.tsv
data/datasets/processed/GGMP/otu.tsv
```

## 8. How to run `gai_cal.py`

The script expects:

```bash
python gai_cal.py <meta.tsv> <otu.tsv> <output_dir>
```

### Run on AGP

From the repository root:

```bash
python gai_cal.py \
  data/datasets/processed/AGP/meta.tsv \
  data/datasets/processed/AGP/otu.tsv \
  output/AGP_gai
```

### Run on GGMP

From the repository root:

```bash
python gai_cal.py \
  data/datasets/processed/GGMP/meta.tsv \
  data/datasets/processed/GGMP/otu.tsv \
  output/GGMP_gai
```

## 9. What `gai_cal.py` produces

For each run, the script writes into the output directory you pass as the third argument.

Typical outputs include:

- tuned model summary
- saved final model
- `adjust_values.tsv`
- `result.tsv`

Main result file examples:

```text
output/AGP_gai/result.tsv
output/GGMP_gai/result.tsv
```

## 10. Notes

- Run all commands from the repository root so the relative paths above work directly.
- `gai_cal.py` is the original age-regression-based GAI pipeline. It predicts age first and then derives GAI.
- If you only want the generated training CSVs, use:
  - `AGP_data.csv`
  - `GGMP_data.csv`
- If you want to regenerate those CSVs, run:

```bash
python create_training_data.py
```
