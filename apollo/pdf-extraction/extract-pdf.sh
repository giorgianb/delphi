#!/usr/bin/bash

if [[ "$#" != "2" ]]; then
    echo "Usage: extract-pdf.sh input_pdf_file.pdf output_csv_file.csv"
    exit 1
fi

INPUT_FILE=$(realpath "$1")
OUTPUT_FILE=$(realpath "$2")

cd "$(dirname "$0")"
mvn exec:java -D exec.mainClass=com.greekengineering.lecturetoc.App -Dexec.args="'$INPUT_FILE' '$OUTPUT_FILE'"
