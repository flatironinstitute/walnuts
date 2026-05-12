#!/usr/bin/env bash
# Run include-what-you-use across the project via compile_commands.json.
#
# Usage:
#   ./run_iwyu.sh                    # run on all translation units
#   ./run_iwyu.sh path/to/file.cpp   # run on a specific file

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
IWYU_PREFIX=$(brew --prefix include-what-you-use)
IWYU_TOOL="$IWYU_PREFIX/bin/iwyu_tool.py"
BUILD_DIR="$PROJECT_ROOT/build"
MAPPING_FILE="$PROJECT_ROOT/iwyu.imp"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "Error: $BUILD_DIR/compile_commands.json not found."
  echo "Run: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  exit 1
fi

# Build a list of --check_also flags, one per project header
CHECK_ALSO=()
while IFS= read -r hpp; do
  CHECK_ALSO+=(-Xiwyu --check_also="$hpp")
done < <(find "$PROJECT_ROOT/include" -name '*.hpp')

"$IWYU_TOOL" \
  -p "$BUILD_DIR" \
  "$@" \
  -- \
  -Xiwyu --mapping_file="$MAPPING_FILE" \
  -Xiwyu --no_fwd_decls \
  -Xiwyu --cxx17ns \
  "${CHECK_ALSO[@]}"
