#!/usr/bin/env bash
# Re-runs the mutation tests recorded in VERIFICATION.md. For each patch in this directory (or
# the patches given as arguments) it copies the sources to a temporary directory, applies the
# patch there (the working tree is never modified), builds the release configuration, runs the
# GoogleTest binary and CTest, and counts failures. A mutation is "caught" when at least one
# test fails. Exits non-zero if any mutation survives.
#
#   tools/mutations/run_mutations.sh [path/to/mutation.patch ...]
#
# Needs cmake, ninja, patch and GoogleTest. JOBS (default 2) sets build parallelism.
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
jobs=${JOBS:-2}
work=$(mktemp -d "${TMPDIR:-/tmp}/int8k-mutations.XXXXXX")
trap 'rm -rf "$work"' EXIT

if [ "$#" -gt 0 ]; then
  patches=("$@")
else
  patches=("$root"/tools/mutations/*.patch)
fi

survivors=0
printf '| mutation | GoogleTest failures | CTest failures (of total) | caught |\n'
printf '|---|---|---|---|\n'
for patch_file in "${patches[@]}"; do
  name=$(basename "$patch_file" .patch)
  src="$work/$name"
  mkdir -p "$src"
  (cd "$root" && tar cf - --exclude='./build*' --exclude='./.git' .) | (cd "$src" && tar xf -)
  patch --silent -d "$src" -p1 < "$patch_file"

  cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release > "$src/configure.log"
  cmake --build "$src/build" -j "$jobs" > "$src/build.log"

  # gtest prints "[  FAILED  ] N tests, listed below:" only when something failed.
  gtest_out=$("$src/build/tests/int8k_tests" 2>&1 || true)
  gtest_failed=$(sed -n 's/^\[  FAILED  \] \([0-9]*\) tests\{0,1\}, listed below:$/\1/p' <<< "$gtest_out")
  gtest_failed=${gtest_failed:-0}
  # (Here-strings, not `printf | grep -q`: with pipefail, grep -q closing the pipe early would
  # make printf fail with SIGPIPE and the check misfire.)
  if ! grep -q '^\[==========\] .* ran\.' <<< "$gtest_out"; then
    gtest_failed="crashed"
  fi

  ctest_out=$(ctest --test-dir "$src/build" -j "$jobs" 2>&1 || true)
  summary=$(grep 'tests passed, ' <<< "$ctest_out" || true)
  ctest_failed=$(sed -n 's/.* \([0-9]*\) tests\{0,1\} failed out of \([0-9]*\)/\1 of \2/p' <<< "$summary")

  caught="yes"
  case "$ctest_failed" in
    "0 of "* | "") caught="NO"; survivors=$((survivors + 1)) ;;
  esac
  printf '| %s | %s | %s | %s |\n' "$name" "$gtest_failed" "${ctest_failed:-?}" "$caught"
  rm -rf "$src"
done

if [ "$survivors" -gt 0 ]; then
  echo "$survivors mutation(s) survived" >&2
  exit 1
fi
