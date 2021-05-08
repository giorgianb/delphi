#!/usr/bin/bash

if [[ "$#" != 2 ]]; then
    echo "Usage: apollo.sh textbook.pdf training_file.csv"
fi

BASE=$(realpath "$(dirname $(dirname $0))")
TMP_OUT=$(mktemp)
TMP_OUT_CLEAN=$(mktemp)

INPUT_FILE=$(realpath "$1")
OUTPUT_FILE=$(realpath "$2")

echo "[Extracting PDF]"
"$BASE/apollo/pdf-extraction/extract-pdf.sh" "$INPUT_FILE" "$TMP_OUT"
cd "$BASE/apollo"
echo "[Cleaning Extracted Output]"
python process.py "$TMP_OUT" "$TMP_OUT_CLEAN"
cd "$BASE/SECTOR/utils"
echo "[Generating Training Data]"
python preprocess_data.py $TMP_OUT_CLEAN "$OUTPUT_FILE"

