#!/usr/bin/env bash
# Setup script for the housecorr3d repository on a SLURM cluster.
# Invoked by `o3b platform setup -p slurm`; environment variables are injected
# by that command from the resolved platform config.
set -euo pipefail

PATH_WS="${PATH_WS:-/work/dlclarge1/sommerl-od3d}"
PATH_CUDA="${PATH_CUDA:-/usr/local/cuda-12.4}"
REPO_URL="${REPO_URL:-}"                                    # housecorr3d SSH URL
REPO_NAME="${REPO_NAME:-$(basename "${REPO_URL}" .git)}"   # e.g. HouseCorr3Dv2
BRANCH="${BRANCH:-main}"
PULL="${PULL:-true}"
PULL_SUBMODULES="${PULL_SUBMODULES:-true}"
SKIP_SUBMODULES="${SKIP_SUBMODULES:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
INSTALL_DIFF3F="${INSTALL_DIFF3F:-false}"
INSTALL_DENSEMATCHER="${INSTALL_DENSEMATCHER:-false}"
DEPS_TAG="${DEPS_TAG:-}"   # e.g. "densematcher" or "densematcher_diff3f"; appended to venv name

export CUDA_HOME="${PATH_CUDA}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CPATH="${CPATH:-}:${CUDA_HOME}/targets/x86_64-linux/include"
export LIBRARY_PATH="${LIBRARY_PATH:-}:${CUDA_HOME}/targets/x86_64-linux/lib"

REPO_PATH="${PATH_WS}/${REPO_NAME}"

# e.g. /usr/local/cuda-12.4 + 3.10 + 2.6.0  →  venv_py310_cu124_torch26
CUDA_TAG="cu$(basename "${PATH_CUDA}" | sed 's/cuda-//;s/\.//')"
PY_TAG="py$(echo "${PYTHON_VERSION}" | sed 's/\.//')"
TORCH_TAG="torch$(echo "${TORCH_VERSION}" | cut -d. -f1,2 | sed 's/\.//')"
VENV_PATH="${REPO_PATH}/venv_${PY_TAG}_${CUDA_TAG}_${TORCH_TAG}${DEPS_TAG:+_${DEPS_TAG}}"

LOCK_FILE="${PATH_WS}/setup_slurm.lock"
exec 200>"${LOCK_FILE}"
echo "--- acquiring setup lock (${LOCK_FILE}) ---"
flock -x 200   # blocks until no other setup_slurm.sh holds the lock

echo "=== housecorr3d slurm setup ==="
echo "  repo  : ${REPO_PATH}"
echo "  branch: ${BRANCH}"
echo "  cuda  : ${CUDA_HOME}"
echo "  venv  : ${VENV_PATH}"

if [ ! -d "${REPO_PATH}" ]; then
    if [ -z "${REPO_URL}" ]; then
        echo "ERROR: ${REPO_PATH} does not exist and REPO_URL is not set."
        exit 1
    fi
    echo "--- git clone ${REPO_URL} ---"
    git clone "${REPO_URL}" "${REPO_PATH}"
fi

cd "${REPO_PATH}"

if [ "${PULL}" = "true" ] || [ "${PULL}" = "True" ]; then
    echo "--- git pull origin ${BRANCH} ---"
    git pull origin "${BRANCH}"
fi

if [ "${PULL_SUBMODULES}" = "true" ] || [ "${PULL_SUBMODULES}" = "True" ]; then
    echo "--- git submodule update ---"
    if [ -n "${GITHUB_TOKEN}" ]; then
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
    fi
    git submodule sync --recursive
    if [ -z "${SKIP_SUBMODULES}" ]; then
        git submodule update --init --recursive
    else
        git submodule init
        for sub in $(git submodule status | awk '{print $2}'); do
            _skip=false
            for s in ${SKIP_SUBMODULES}; do
                [ "$sub" = "$s" ] && _skip=true && break
            done
            if [ "$_skip" = "false" ]; then
                git submodule update --init --recursive -- "$sub"
            fi
        done
    fi
fi

if [ ! -d "${VENV_PATH}" ]; then
    echo "--- python${PYTHON_VERSION} -m venv ${VENV_PATH} ---"
    "python${PYTHON_VERSION}" -m venv "${VENV_PATH}"
fi
echo "--- activate ${VENV_PATH} ---"
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

echo "--- pip bootstrap ---"
pip install --upgrade pip setuptools wheel

# ensure correct version
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

echo "--- pip install o3b ---"
pip install -e third_party/o3b --no-build-isolation

echo "--- pip install housecorr3d ---"
pip install -e .

echo "--- pip install optional deps ---"
pip install pyrender2
pip install xatlas


if [ "${INSTALL_DIFF3F}" = "true" ] || [ "${INSTALL_DIFF3F}" = "True" ]; then
    echo "--- pip install diff3f deps ---"
    # Always install pytorch3d pre-built wheel matching the torch+cuda combo.
    # e.g. TORCH_VERSION=2.6.0, CUDA_TAG=cu124  →  pytorch3d==0.7.8+pt2.6.0cu124
    P3D_TORCH_TAG="pt${TORCH_VERSION}${CUDA_TAG}"
    echo "--- pip install pytorch3d==0.7.8+${P3D_TORCH_TAG} ---"
    pip install "pytorch3d==0.7.8+${P3D_TORCH_TAG}" \
        --extra-index-url https://miropsota.github.io/torch_packages_builder
    pip install diffusers transformers accelerate
    pip install git+https://github.com/skoch9/meshplot.git
    pip install pythreejs
    # Recompile gdist from source against the final numpy in this venv.
    # Binary wheels are pinned to the numpy ABI at build time; optional deps
    # (pytorch3d, diffusers, …) can silently shift numpy afterwards, causing
    # "numpy.dtype size changed, may indicate binary incompatibility".
    pip install --no-binary gdist gdist --force-reinstall --no-cache-dir
fi


if [ "${INSTALL_DENSEMATCHER}" = "true" ] || [ "${INSTALL_DENSEMATCHER}" = "True" ]; then
  # pip install --no-cache-dir xformers  # requires torch 2.10 not required
  pip install --no-cache-dir robust-laplacian
  pip install --no-cache-dir potpourri3d
  pip install diffusers[torch]==0.27.2
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/Mask2Former # export CUDA_HOME="/usr/local/cuda-12.4" & 
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/ODISE
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/stablediffusion
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/featup
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/dift
  pip install "numpy<2.0" # ==1.24.1       # < 2.0: numpy.core.multiarray removed in NumPy 2
  pip install "Pillow<10.0.0"    # Image.LINEAR removed in Pillow 10
  pip install setuptools==81.0.0
  pip install pytorch-lightning==1.9.5 kornia==0.7.2 pillow==9.3.0 transformers==4.27.0 matplotlib==3.9.3
  pip install huggingface-hub==0.25.2
fi

echo "=== setup complete ==="
