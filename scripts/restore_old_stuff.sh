#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_ROOT="${1:-${ROOT}/old_stuff/tesseract2026_noncomparison_20260806}"
MANIFEST="${ROOT}/ARCHIVE_MANIFEST.txt"

[[ -d "${ARCHIVE_ROOT}" ]] || {
  echo "Archive does not exist: ${ARCHIVE_ROOT}" >&2
  exit 1
}

while IFS= read -r relative; do
  [[ -n "${relative}" && "${relative}" != \#* ]] || continue
  source_path="${ARCHIVE_ROOT}/${relative}"
  destination="${ROOT}/${relative}"
  [[ -e "${source_path}" ]] || {
    echo "Missing archived path: ${source_path}" >&2
    exit 1
  }
  [[ ! -e "${destination}" ]] || {
    echo "Refusing to overwrite existing path: ${destination}" >&2
    exit 1
  }
done < "${MANIFEST}"

while IFS= read -r relative; do
  [[ -n "${relative}" && "${relative}" != \#* ]] || continue
  source_path="${ARCHIVE_ROOT}/${relative}"
  destination="${ROOT}/${relative}"
  mkdir -p "$(dirname "${destination}")"
  mv -- "${source_path}" "${destination}"
done < "${MANIFEST}"

echo "Restored archived files to ${ROOT}"
