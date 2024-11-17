#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

combine_md ./src '*.py' src_files.md
combine_md ./examples '*.py' examples_files.md
combine_md ./tests '*.py' tests_files.md
cat src_files.md examples_files.md tests_files.md > all_files.md
echo "Combined source files: all_files.md"
