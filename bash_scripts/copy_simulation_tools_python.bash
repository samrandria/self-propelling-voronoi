#!usr/bin/env bash

SIMULATION_SOURCES_DIR="../simulation_sources"
PYTHON_DIR="python_files"
BASH_DIR="bash_files"
CYTHON_DIR="cython_files"

for file in "$SIMULATION_SOURCES_DIR/$PYTHON_DIR/"*
do
	echo "Copying Python file : $file"
	cp "$file" ./
done

for file in "$SIMULATION_SOURCES_DIR/$BASH_DIR/"*
do
	echo "Copying Bash file : $file"
	cp "$file" ./
done

# for file in "$SIMULATION_SOURCES_DIR/$CYTHON_DIR/"*
# do
# 	echo "Copying Cython file : $file"
# 	cp "$file" ./
# done


