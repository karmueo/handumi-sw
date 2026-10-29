#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_XRT=1
TRACKING_DEVICE="vive"
ROBOT=""
WITH_SIM=0
WITH_CUDA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-xrt) SKIP_XRT=1 ;;
    --device)
      shift
      TRACKING_DEVICE="${1:-}"
      case "$TRACKING_DEVICE" in
        vive|meta) SKIP_XRT=1 ;;
        pico) SKIP_XRT=0 ;;
        *) echo "error: --device expects vive, meta or pico" >&2; exit 1 ;;
      esac
      ;;
    --sim) WITH_SIM=1 ;;
    --cuda) WITH_CUDA=1 ;;
    --robot)
      shift
      if [[ $# -eq 0 || ( "$1" != "piper" && "$1" != "openarmv1" ) ]]; then
        echo "error: --robot expects piper or openarmv1" >&2
        exit 1
      fi
      ROBOT="$1"
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "       usage: $0 [--device vive|meta|pico] [--skip-xrt] [--sim] [--cuda] [--robot piper|openarmv1]" >&2
      exit 1
      ;;
  esac
  shift
done

XROBO_DIR="external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
SIBLING_XROBO="$ROOT/../GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
XROBO_REPO="https://github.com/leoperezz/XRoboToolkit-PC-Service-Pybind.git"
XROBO_PC_SERVICE_REPO="https://github.com/leoperezz/XRoboToolkit-PC-Service"
XRT_PC_SERVICE_SCRIPT="/opt/apps/roboticsservice/runService.sh"
XRT_PC_SERVICE_DEB="XRoboToolkit_PC_Service_1.0.1.0_ubuntu_22.04_amd64.deb"
XRT_PC_SERVICE_URL="${XROBO_PC_SERVICE_REPO}/raw/main/releases/download/v1.0.1.0/${XRT_PC_SERVICE_DEB}"
ARCH="$(uname -m)"
VENV_DIR=".venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

activate_venv() {
  export VIRTUAL_ENV="$ROOT/$VENV_DIR"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  unset PYTHONHOME
}

pip_installed() {
  uv pip show "$1" >/dev/null 2>&1
}

ensure_rig_config() {
  if [[ -f "configs/rig.yaml" ]]; then
    echo "==> Reusing local rig configuration (configs/rig.yaml)"
    return 0
  fi
  if [[ "$TRACKING_DEVICE" == "vive" ]]; then
    cp configs/rig.example.yaml configs/rig.yaml
  else
    sed "s/device: meta/device: $TRACKING_DEVICE/" configs/rig.legacy.example.yaml > configs/rig.yaml
  fi
  echo "==> Created configs/rig.yaml from the example; edit it for this machine"
}

# ── XRoboToolkit sources ──────────────────────────────────────────────────────

ensure_xrobotoolkit_sources() {
  if [[ -d "$XROBO_DIR" ]]; then
    echo "==> XRoboToolkit sources already present, skipping fetch"
    return 0
  fi

  mkdir -p external_dependencies

  if [[ -d "$SIBLING_XROBO" ]]; then
    echo "==> Copying XRoboToolkit dependency from GR00T-WholeBodyControl"
    cp -a "$SIBLING_XROBO" "$XROBO_DIR"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "error: missing dependency directory: $XROBO_DIR" >&2
    echo "       Install git or copy the dependency manually into external_dependencies/." >&2
    exit 1
  fi

  echo "==> Cloning XRoboToolkit-PC-Service-Pybind"
  git clone "$XROBO_REPO" "$XROBO_DIR"
}

ensure_xrobotoolkit_native_lib() {
  if [[ "$ARCH" == "x86_64" ]] && [[ -f "$XROBO_DIR/lib/libPXREARobotSDK.so" ]]; then
    echo "==> PXREARobotSDK x86_64 library already built, skipping"
    return 0
  fi

  if [[ "$ARCH" == "aarch64" ]] && [[ -f "$XROBO_DIR/lib/aarch64/libPXREARobotSDK.so" ]]; then
    echo "==> PXREARobotSDK aarch64 library already built, skipping"
    return 0
  fi

  if [[ "$ARCH" == "x86_64" ]]; then
    echo "==> Building PXREARobotSDK for x86_64"
    XRT_TMP="$XROBO_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [[ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]]; then
      git clone "$XROBO_PC_SERVICE_REPO.git" "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" >/dev/null
    bash build.sh
    popd >/dev/null
    mkdir -p "$XROBO_DIR/lib" "$XROBO_DIR/include"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
      "$XROBO_DIR/include/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
      "$XROBO_DIR/include/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
      "$XROBO_DIR/lib/"
    rm -rf "$XRT_TMP"
    echo "==> PXREARobotSDK x86_64 native library built and installed"
    return 0
  fi

  if [[ "$ARCH" == "aarch64" ]]; then
    echo "==> Building PXREARobotSDK for aarch64"
    XRT_TMP="$XROBO_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [[ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]]; then
      git clone -b orin "$XROBO_PC_SERVICE_REPO.git" "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" >/dev/null
    bash build.sh
    popd >/dev/null
    mkdir -p "$XROBO_DIR/lib/aarch64" "$XROBO_DIR/include/aarch64"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
      "$XROBO_DIR/include/aarch64/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
      "$XROBO_DIR/include/aarch64/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
      "$XROBO_DIR/lib/aarch64/"
    rm -rf "$XRT_TMP"
    echo "==> PXREARobotSDK aarch64 native library built and installed"
    return 0
  fi

  echo "error: unsupported architecture for XRoboToolkit: $ARCH" >&2
  exit 1
}

ensure_xrobotoolkit_pc_service() {
  if [[ -f "$XRT_PC_SERVICE_SCRIPT" ]]; then
    echo "==> XRoboToolkit PC service already installed"
    return 0
  fi

  if [[ "$ARCH" != "x86_64" ]]; then
    echo "==> Skipping XRoboToolkit PC service auto-install on $ARCH"
    echo "    Build or install the package manually from $XROBO_PC_SERVICE_REPO"
    return 0
  fi

  if ! command -v dpkg >/dev/null 2>&1; then
    echo "warning: dpkg not found; install XRoboToolkit PC service manually" >&2
    echo "         $XRT_PC_SERVICE_URL" >&2
    return 0
  fi

  echo "==> Installing XRoboToolkit PC service ($XRT_PC_SERVICE_DEB)"
  XRT_DEB_TMP="$(mktemp -d)"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$XRT_DEB_TMP/$XRT_PC_SERVICE_DEB" "$XRT_PC_SERVICE_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$XRT_DEB_TMP/$XRT_PC_SERVICE_DEB" "$XRT_PC_SERVICE_URL"
  else
    rm -rf "$XRT_DEB_TMP"
    echo "error: curl or wget required to download XRoboToolkit PC service" >&2
    exit 1
  fi

  if ! sudo dpkg -i "$XRT_DEB_TMP/$XRT_PC_SERVICE_DEB"; then
    sudo apt-get install -f -y
    sudo dpkg -i "$XRT_DEB_TMP/$XRT_PC_SERVICE_DEB"
  fi
  rm -rf "$XRT_DEB_TMP"
}

# ── Venv + project deps ───────────────────────────────────────────────────────

ensure_venv() {
  if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    echo "==> Reusing existing virtual environment ($VENV_DIR)"
  else
    echo "==> Creating uv virtual environment (Python >= 3.12)"
    uv venv --python ">=3.12"
  fi
  # Activate for all subsequent uv pip commands in this session.
  activate_venv
}

ensure_project_deps() {
  echo "==> Syncing project dependencies (update only if needed)"
  local extras=()
  [[ "$TRACKING_DEVICE" == "vive" ]] && extras+=(--extra vive)
  [[ "$WITH_SIM" -eq 1 ]] && extras+=(--extra sim)
  [[ "$WITH_CUDA" -eq 1 ]] && extras+=(--extra cuda)
  [[ -n "$ROBOT" ]] && extras+=(--extra "${ROBOT/openarmv1/openarm}")
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" uv sync "${extras[@]}"
}

ensure_shell_completion() {
  local activate_file="$VIRTUAL_ENV/bin/activate"
  local activate_fish="$VIRTUAL_ENV/bin/activate.fish"
  local marker="# HandUMI shell completion"

  if [[ -f "$activate_file" ]] && ! grep -Fq "$marker" "$activate_file"; then
    printf '\n%s\n' "$marker" >> "$activate_file"
    printf '%s\n' 'if [ -n "${BASH_VERSION:-}" ]; then' >> "$activate_file"
    printf '%s\n' '  eval "$(handumi completion bash)"' >> "$activate_file"
    printf '%s\n' 'elif [ -n "${ZSH_VERSION:-}" ]; then' >> "$activate_file"
    printf '%s\n' '  eval "$(handumi completion zsh)"' >> "$activate_file"
    printf '%s\n' 'fi' >> "$activate_file"
  fi

  if [[ -f "$activate_fish" ]] && ! grep -Fq "$marker" "$activate_fish"; then
    printf '\n%s\n' "$marker" >> "$activate_fish"
    printf '%s\n' 'handumi completion fish | source' >> "$activate_fish"
  fi
  echo "==> Installed HandUMI shell completion for the virtual environment"
}

ensure_openarm_system_deps() {
  [[ "$ROBOT" == "openarmv1" ]] || return 0
  if command -v openarm-can-cli >/dev/null 2>&1 && \
     dpkg-query -W -f='${Status}' libopenarm-can-dev 2>/dev/null | grep -q "install ok installed"; then
    echo "==> OpenArm CAN system library and tools already installed"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: automatic OpenArm system setup currently supports Ubuntu/Debian" >&2
    exit 1
  fi
  echo "==> Installing official OpenArm CAN system packages"
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:openarm/main
  sudo apt-get update
  sudo apt-get install -y libopenarm-can-dev openarm-can-utils
}

# ── XRoboToolkit Python package ───────────────────────────────────────────────

ensure_xrobotoolkit_python_package() {
  if pip_installed xrobotoolkit-sdk || pip_installed xrobotoolkit_sdk; then
    echo "==> xrobotoolkit_sdk already installed, skipping rebuild"
    return 0
  fi

  echo "==> Installing XRoboToolkit SDK (editable, no build isolation)"
  # cmake and pybind11 are declared in pyproject.toml so uv sync already put
  # them in the venv before we reach this point.
  export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
  uv pip install --no-build-isolation -e "$XROBO_DIR/"
}

# ── Main ──────────────────────────────────────────────────────────────────────

ensure_rig_config
if [[ "$SKIP_XRT" -eq 1 ]]; then
  echo "==> --skip-xrt: skipping XRoboToolkit (PICO) sources/build/package"
else
  ensure_xrobotoolkit_sources
  ensure_xrobotoolkit_native_lib
  ensure_xrobotoolkit_pc_service
fi
ensure_venv
ensure_openarm_system_deps
ensure_project_deps
ensure_shell_completion
if [[ "$SKIP_XRT" -ne 1 ]]; then
  ensure_xrobotoolkit_python_package
fi

echo "==> Installation complete"
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo ""

# ── Runtime note: XRoboToolkit service ───────────────────────────────────────
if [[ "$TRACKING_DEVICE" == "vive" ]]; then
  echo "NOTE: Start SteamVR before using Vive. Run handumi calibrate umi --help for calibration."
elif [[ "$SKIP_XRT" -eq 1 ]]; then
  echo "NOTE: XRoboToolkit (PICO) was skipped (--skip-xrt). Using Meta Quest"
  echo "      tracking needs no PC service — see docs/README_quest.md."
else
  if [[ -f "$XRT_PC_SERVICE_SCRIPT" ]]; then
    echo "NOTE: XRoboToolkit PC service found at $XRT_PC_SERVICE_SCRIPT"
    echo "      Start it before running any xrobotoolkit_sdk scripts:"
    echo "        bash $XRT_PC_SERVICE_SCRIPT"
  else
    echo "WARNING: XRoboToolkit PC service not found at $XRT_PC_SERVICE_SCRIPT"
    echo "         xrt.init() will crash (core dump) if the service is not running."
    echo "         Install the XRoboToolkit PC service from:"
    echo "           $XROBO_PC_SERVICE_REPO"
    echo "         or download:"
    echo "           $XRT_PC_SERVICE_URL"
  fi
fi
