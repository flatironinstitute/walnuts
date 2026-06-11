#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
IWYU_PREFIX=$(brew --prefix include-what-you-use)
IWYU_TOOL="$IWYU_PREFIX/bin/iwyu_tool.py"
BUILD_DIR="$PROJECT_ROOT/build"
MAPPING_FILE="$PROJECT_ROOT/iwyu/iwyu.imp"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found."
  echo "Run: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  exit 1
fi

# Build --check_also flags for all project headers, excluding umbrella
CHECK_ALSO=()
while IFS= read -r hpp; do
  if [[ "$hpp" != *"include/walnuts.hpp" ]]; then
    CHECK_ALSO+=(-Xiwyu --check_also="$hpp")
  fi
done < <(find "$PROJECT_ROOT/include" -name '*.hpp')

# Resolve file arguments to absolute paths
TARGETS=()
for arg in "$@"; do
  if [[ -f "$PROJECT_ROOT/$arg" ]]; then
    TARGETS+=("$PROJECT_ROOT/$arg")
  elif [[ -f "$arg" ]]; then
    TARGETS+=("$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")")
  else
    echo "Warning: '$arg' not found relative to project root or current directory"
  fi
done

# Build --check_also flags for all project headers
CHECK_ALSO=()
while IFS= read -r hpp; do
  CHECK_ALSO+=(-Xiwyu --check_also="$hpp")
done < <(find "$PROJECT_ROOT/include" -name '*.hpp')

"$IWYU_TOOL" \
  -p "$BUILD_DIR" \
  "${TARGETS[@]}" \
  -- \
  -Xiwyu --mapping_file="$MAPPING_FILE" \
  -Xiwyu --no_fwd_decls \
  -Xiwyu --cxx17ns \
  "${CHECK_ALSO[@]}"
