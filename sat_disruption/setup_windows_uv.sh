#!/usr/bin/env bash
# Reproducible Windows setup using uv (no MSVC / no conda toolchain needed).
#
# Why this is not a plain `uv pip install gala`:
#   * gala publishes NO Windows wheels on PyPI for modern versions, so
#     `uv pip install gala` tries to COMPILE from source (needs MSVC + GSL).
#   * The only prebuilt Windows wheels that exist are gala==1.4.1 (cp37/cp38).
#   * That wheel's C extensions dynamically link `gsl-25.dll`, which is not on
#     PyPI, so we fetch GSL's runtime DLLs from conda-forge ONCE and stage them
#     inside the venv, registered via a .pth `os.add_dll_directory` hook.
#
# Result: a pure-uv Python environment whose only "borrowed" pieces are native
# GSL DLLs copied into the venv (self-contained; the conda temp env is removed).
#
# Run from the sat_disruption/ directory:  bash setup_windows_uv.sh
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION=3.8            # the gala 1.4.1 Windows wheel targets cp37/cp38
VENV=.venv
GSL_STAGE="$VENV/gsl_dlls"

echo "[1/5] creating uv venv (Python $PY_VERSION)"
python -m uv venv "$VENV" --python "$PY_VERSION"

echo "[2/5] installing gala 1.4.1 (prebuilt wheel) + deps"
python -m uv pip install --python "$VENV" --only-binary=gala \
    "gala==1.4.1" h5py matplotlib pytest pyyaml

echo "[3/5] fetching GSL runtime DLLs from conda-forge (temp env)"
conda create -n satdis_gsltmp --override-channels -c conda-forge gsl -y
GSLBIN="$(conda run -n satdis_gsltmp python -c "import sys,os;print(os.path.join(sys.prefix,'Library','bin'))")"

echo "[4/5] staging DLL closure into $GSL_STAGE"
mkdir -p "$GSL_STAGE"
cp "$GSLBIN"/*.dll "$GSL_STAGE"/
# gala's wheel imports the literal name 'gsl-25.dll'; conda-forge ships a newer
# soname (gsl-28.dll). The loader matches by filename then resolves gsl_*
# symbols by name, so the newer GSL satisfies the older wheel.
cp "$GSLBIN"/gsl-*.dll "$GSL_STAGE"/gsl-25.dll
# register the stage dir on the DLL search path at interpreter startup
SP="$(./$VENV/Scripts/python.exe -c "import site;print([p for p in site.getsitepackages() if p.endswith('site-packages')][0])")"
printf '%s\n' "import os, sys; _p=os.path.join(sys.prefix, 'gsl_dlls'); os.path.isdir(_p) and os.add_dll_directory(_p)" > "$SP/gala_gsl_dlls.pth"

echo "[5/5] removing temp conda env and smoke-testing the import"
conda env remove -n satdis_gsltmp -y
./$VENV/Scripts/python.exe -c "import gala; from gala.dynamics.nbody import DirectNBody; print('gala', gala.__version__, 'OK')"

echo
echo "Done. Run the pipeline with:"
echo "  ./$VENV/Scripts/python.exe run_simulation.py --config config.yaml"
echo "  ./$VENV/Scripts/python.exe -m pytest tests -q"
