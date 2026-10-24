#!usr/bin/env bash


# Export ?

# Add the 'Agg' instruction to flocking_confluent_marchetti to run it on the server without a display
FILE_FOR_AGG_ADD="flocking_confluent_marchetti.py"
CYTHON_FILES_DIR="./"
C_FILES_DIR="c_files"


# Create a cython_files directory if it does not exist
mkdir -p "$CYTHON_FILES_DIR"

# Copy all the *.py files inside it
cp -t "$CYTHON_FILES_DIR"/ *.py

# Move into the newly created dir
cd "$CYTHON_FILES_DIR"

sed -i "/import matplotlib/a matplotlib.use('Agg')" "${FILE_FOR_AGG_ADD}"


# Convert every .py file into .pyx EXCEPT for setup.py, parameters_generator.py and main_spv_script.py
for file in *.py 
do
	# Check if we manipulate a file and if this is not one of the .py file we want to keep
	echo "This is $file"
	if [[ -f "$file"  && (  "$file" != "parameters_generator.py" && "$file" != "main_spv_script.py" && "$file" != "setup.py" && "$file" != "script_profiler.py" && "$file" != "setup_cython_profiler.py" ) ]]  
	then
		# name without extension (escape the dot because otherwise it is the "local_dir" character
		name=${file%\.*}
		echo ${name}
		mv "$file" "$name".pyx
	fi ;
done

# Compile using setup.py

python3 setup.py build_ext --inplace


# Create a build/c_files if it does not exist and move the .c files inside it
mkdir -p "build/$C_FILES_DIR"
mv -t "build/$C_FILES_DIR" *.c

