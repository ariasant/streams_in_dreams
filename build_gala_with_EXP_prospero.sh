#!/bin/bash
# Build pyEXP + a gala fork with EXP support, on this cluster (aridata1 / "prospero").
#
# This is the aridata1 counterpart to build_gala_with_EXP.sh (which targets a different HPC
# facility's module set and is not usable here). Module choices and the Eigen3/ninja/doxygen/uv
# workarounds below were derived by checking `module avail` on this machine against EXP-code's
# actual CMakeLists.txt (github.com/EXP-code/EXP, devel branch):
#   - Eigen3 is a hard `find_package(Eigen3 3.4...<3.5 REQUIRED)` and isn't available as a module
#     or system package here, so we vendor the header-only 3.4 branch ourselves below. That
#     find_package call runs in CMake config mode (it wants Eigen3Config.cmake), which cloning
#     the source alone doesn't produce -- Eigen still needs a trivial configure+install (no
#     compilation, just header copying + config-file generation) into a local prefix.
#   - yaml-cpp / pybind11 / HighFive / QuickDigest5 are EXP git submodules (fetched by
#     `--recursive`) -- not system dependencies, nothing to do for those.
#   - TIRPC is optional in EXP's CMakeLists (has its own fallback); this cluster has the libtirpc
#     runtime but not the -devel headers, so TIRPC_FOUND will likely be false. That's expected --
#     check the cmake configure log below only if the build itself fails.
#   - ninja/doxygen/uv aren't available here and aren't EXP requirements, just the other script's
#     tool choices -- substituted with "Unix Makefiles", -DBUILD_DOCS=NO, and plain venv+pip.
#   - The fftw3_double module sets a nonstandard env var (FFTW3_DOUBLEDIR) that EXP's
#     cmake/FindFFTW.cmake doesn't look for (it checks -DFFTW_ROOT=, $FFTWDIR, or pkg-config) --
#     passed through explicitly as -DFFTW_ROOT below.
#   - EL8's default compiler (compilers/gcc/system, GCC 8.5.0) keeps std::filesystem in a separate
#     libstdc++fs.a (only merged into libstdc++ itself in GCC 9+), and EXP's CMakeLists doesn't
#     link it explicitly. Passing -lstdc++fs via CMAKE_*_LINKER_FLAGS "fixes" it for whichever
#     target CMake happens to place the flag ahead of the referencing objects for, and breaks it
#     for whichever target it doesn't (GNU ld resolves a static archive in one left-to-right pass,
#     so a library placed before anything references it gets skipped) -- it's not worth chasing
#     per-target. compilers/gcc/12.2.0 (also a module here) side-steps the whole class of problem:
#     GCC 9+ folds std::filesystem into libstdc++.so itself, so nothing needs -lstdc++fs at all.
#   - -DENABLE_PYEXP_ONLY=YES (EXP's own documented metaflag for "just the Python bindings") is
#     kept anyway even though it's no longer required to avoid the above -- fewer targets, faster
#     build, and we only ever needed pyEXP.
#   - gala's devel branch requires Python >=3.11, newer than the python3.10.5 module the rest of
#     this repo (perturbed_host/) uses -- so this environment uses python3.12.7 instead. pyEXP is
#     a compiled extension tied to one Python's ABI (pyEXP.cpython-310-*.so only imports under
#     3.10), so EXP itself must be built against 3.12.7 too, not just the expgala venv.
#
# Run from the directory you want EXP/eigen-3.4/gala/expgala/ created in (e.g. repo root).

set -euo pipefail

module purge
module load apps/cmake/3.25.1/gcc-8.5.0 \
            compilers/gcc/12.2.0 \
            mpi/openmpi/4.1.5/gcc-8.5.0 \
            apps/hdf5_serial/1.14.4/gcc-8.5.0 \
            libs/fftw3_double/3.3.8/gcc-8.5.0+openmpi-4.1.5 \
            apps/python3/3.12.7/gcc-8.5.0

BASE=$PWD

# --- Vendor Eigen3 (header-only; no module/system package on this cluster) -------------------
if [ ! -d "$BASE/eigen-3.4" ]; then
    git clone --branch 3.4 --depth 1 https://gitlab.com/libeigen/eigen.git "$BASE/eigen-3.4"
fi
if [ ! -f "$BASE/eigen-3.4-install/share/eigen3/cmake/Eigen3Config.cmake" ]; then
    # BUILD_TESTING/EIGEN_BUILD_* OFF: skip Eigen's own (huge) test suite and the optional
    # BLAS/LAPACK compatibility shims -- we only need the headers + Eigen3Config.cmake.
    cmake -G "Unix Makefiles" -S "$BASE/eigen-3.4" -B "$BASE/eigen-3.4/build" \
        --install-prefix "$BASE/eigen-3.4-install" \
        -DBUILD_TESTING=OFF -DEIGEN_BUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF \
        -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF
    cmake --build "$BASE/eigen-3.4/build"
    cmake --install "$BASE/eigen-3.4/build"
fi

# --- Download EXP (recursive: also fetches yaml-cpp/pybind11/HighFive/QuickDigest5) -----------
if [ ! -d "$BASE/EXP" ]; then
    git clone --recursive https://github.com/EXP-code/EXP.git -b devel "$BASE/EXP"
fi

# --- Build EXP ----------------------------------------------------------------------------------
cd "$BASE/EXP"
cmake -G "Unix Makefiles" -B build \
 -DCMAKE_INSTALL_RPATH="$PWD/install/lib" \
 --install-prefix "$PWD/install" \
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_DOCS=NO \
 -DENABLE_PYEXP_ONLY=YES \
 -DCMAKE_PREFIX_PATH="$BASE/eigen-3.4-install" \
 -DFFTW_ROOT="$FFTW3_DOUBLEDIR"
cmake --build build -j "$(nproc)"
cmake --install build

EXP_path="$PWD"

# --- Create virtual environment ------------------------------------------------------------------
cd "$BASE"
venv_name="expgala"
python3 -m venv "$venv_name"
source "$venv_name/bin/activate"
pip install --upgrade pip

# Covers everything actually imported across plummer_bfe_report.ipynb / EXP4DREAMS.py /
# sampling_utils.py / EXP_visual_fns.py. scipy>=1.15 is required for scipy.differentiate.derivative
# (used in sampling_utils.py), newer than the root repo's requirements.txt pin.
pip install "numpy>=1.24,<2.0" "scipy>=1.15" h5py pyyaml matplotlib astropy corner pandas \
            imageio pillow pynbody tqdm k3d similaritymeasures

# --- Download and build gala (devel branch, with EXP support) -----------------------------------
if [ ! -d "$BASE/gala" ]; then
    git clone -b devel https://github.com/adrn/gala.git "$BASE/gala"
fi
cd "$BASE/gala"

# Needs to be set whenever this venv is activated to use gp.EXPPotential (not just at build time).
export GALA_EXP_PREFIX="$EXP_path/install"

# gala's cyexp.pyx bridges directly to EXP's headers (via GALA_EXP_PREFIX, added to its own -I
# flags), but those headers in turn `#include <Eigen/Eigen>` etc. without bundling them -- gala's
# build tries pkg-config for eigen3/hdf5/mpi to fill the gap, which isn't set up on this cluster
# (module avail has no eigen3, and neither the hdf5_serial nor openmpi module exports a pkg-config
# file), so it silently drops those include paths instead of failing loudly. CPATH is the
# compiler-level equivalent -- g++ appends it regardless of what -I flags the build system itself
# passes -- so it works without needing gala's build to know about any of this.
export CPATH="$BASE/eigen-3.4-install/include/eigen3:$HDF5_SERIALINCLUDE:$MPI_HOME/include${CPATH:+:$CPATH}"
pip install -ve .

# --- Wire up the venv so plain `source expgala/bin/activate` is enough later -------------------
# Four things the venv's own python3 binary and pyEXP/gala need at *runtime*, not just at build
# time, none of which "source activate" sets up by itself:
#  1. LD_LIBRARY_PATH for libpython3.12.so.1.0 -- expgala/bin/python3 is linked against the
#     module-loaded interpreter, but a fresh shell won't have that module's lib dir on
#     LD_LIBRARY_PATH unless the module is loaded, so `python` inside the venv fails to even start.
#  2. LD_LIBRARY_PATH for GCC 12.2.0's libstdc++.so.6 -- pyEXP/gala were compiled with it (needed
#     for GLIBCXX symbol versions the system's own /lib64/libstdc++.so.6, from GCC 8.5.0, lacks),
#     but a fresh shell resolves libstdc++.so.6 from the system path unless this is set too.
#  3. PYTHONPATH for pyEXP's compiled module (installs under EXP's own prefix, not into the venv).
#  4. GALA_EXP_PREFIX for gp.EXPPotential.
# Appending them to activate (instead of just printing instructions) means future sessions don't
# need to remember any of this or `module load` first.
PY_MM=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MARKER="# --- build_gala_with_EXP_prospero.sh runtime env ---"
if ! grep -qF "$MARKER" "$BASE/expgala/bin/activate"; then
    cat >> "$BASE/expgala/bin/activate" <<EOF

$MARKER
export LD_LIBRARY_PATH="$PYTHONLIB:$GCCDIR/lib64:$GCCDIR/lib:\$LD_LIBRARY_PATH"
export GALA_EXP_PREFIX="$EXP_path/install"
export PYTHONPATH="$EXP_path/install/lib/python${PY_MM}/site-packages:\$PYTHONPATH"
EOF
fi

echo "Done. To use this environment: source $BASE/expgala/bin/activate"
