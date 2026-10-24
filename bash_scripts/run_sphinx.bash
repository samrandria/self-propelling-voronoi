#!usr/bin/env bash

# Sets the PYTHONPATH to the doc directory of the current project and runs sphinx
# Assumes that the Makefile sits at the top of the doc directory.

DOC_DIR='./doc/'

export PYTHONPATH="$(pwd)"
#echo "$(pwd)"

cd "$DOC_DIR"
#echo "$(pwd)"

make html
