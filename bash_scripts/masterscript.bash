#!/usr/bin/bash

PYTHON_COMMAND="/opt/anaconda3/bin/python3"
PBS_TEMPLATE="samplepbs_template.bash"
PBS_NAME_DESIRED="python_pbs"
MAIN_PYTHON_FILE="main_spv_script.py"

# Select the parameter to explore and the range
PARAMETER_TO_EXPLORE="boxsize"
MIN_PARAMETER=10
MAX_PARAMETER=16
STEP_SIZE=1

PARAMETER_FILENAME="parameter_${PARAMETER_TO_EXPLORE}_min_${MIN_PARAMETER}_max_${MAX_PARAMETER}_step_${STEP_SIZE}.txt"

# Call the python script to generate the parameters list
# parameters_generator.py PARAMETER_TO_EXPLORE MIN MAX STEP
"$PYTHON_COMMAND" parameters_generator.py "$PARAMETER_TO_EXPLORE" "$MIN_PARAMETER" "$MAX_PARAMETER" "$STEP_SIZE" 

# Get number of line in the created file to create as many qsubs as needed
NUM_LINE=$(wc -l < "${PARAMETER_FILENAME}")

# Make matplotlib use the 'Agg' to avoid invalid DISPLAY error (this is currently done by the build cython script)
# sed -i "/import matplotlib/a matplotlib.use('Agg')" "${MAIN_PYTHON_FILE}"



# Initialize counter
CTR=0
while read -r line 
do
	# Copy the template
	cp  "${PBS_TEMPLATE}" "${PBS_NAME_DESIRED}_${CTR}"
	# Append the job call line after a pattern "MATCH"
	JOB_CALL_LINE="$line"
	sed -i "/MATCH/a ${JOB_CALL_LINE}" "${PBS_NAME_DESIRED}_${CTR}"
	# Increment counter
	CTR="$(( $CTR+1 ))"
done < "$PARAMETER_FILENAME"

# Call the job manager in background for every file created
for (( i=0; i<NUM_LINE; i++ ))
do
	qsub "${PBS_NAME_DESIRED}_${i}" 
done
