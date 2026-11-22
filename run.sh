#!/bin/sh
# Entry point. Runs with no arguments; exit 0 means "done, check my output".
set -eu
: "${DATA_DIR:=/data}" "${OUTPUT_DIR:=/output}" "${SCRATCH_DIR:=/scratch}"
export DATA_DIR OUTPUT_DIR SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR" HOME="$SCRATCH_DIR"
# Ask glibc malloc for transparent huge pages on big buffers (fewer page faults).
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
exec /app/sessions
