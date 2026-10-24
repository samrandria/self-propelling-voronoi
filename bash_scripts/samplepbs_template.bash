#!/bin/bash

## settings (can be modified)
#PBS -j oe	#(for standard and error outputs)

#PBS -N voronoi	#(Job name)
#PBS -q long	#(Queue name - "standard": default, upto (default) 24 hours, max 36 jobs, "short": upto (default and max) 30 mins but with higher priority, max 36 jobs, "long": upto (default) 3 weeks, max 12 jobs)

##(Note: "#PBS" is a set of word here. Only the first # in "##PBS" means the comment-out, and you can uncomment the command by replacing "##" by "#".)

##PBS -m bea #(optional)(to mail alerts at start, end and abortion of execution)

#PBS -l nodes=satsuki  #(Name of the node on which the job will be run)
##PBS -l nodes=1:ppn=1	#(Number of computing nodes, and number of cores per node used for the job)

##PBS -l walltime=24:00:00 #(Max wallclock time (Max run time), hh:mm:ss)


## DO NOT TOUCH 
cd $PBS_O_WORKDIR #(to move into NFS directory (/nfshome))

## DO NOT TOUCH
echo ------------------------------------------------------  > job${PBS_JOBID%%.*}.log 
echo -n 'Job is running on node '                            >> job${PBS_JOBID%%.*}.log 
cat $PBS_NODEFILE                                            >> job${PBS_JOBID%%.*}.log 
echo ------------------------------------------------------  >> job${PBS_JOBID%%.*}.log 
echo PBS: qsub is running on $PBS_O_HOST                     >> job${PBS_JOBID%%.*}.log 
echo PBS: originating queue is $PBS_O_QUEUE                  >> job${PBS_JOBID%%.*}.log 
echo PBS: executing queue is $PBS_QUEUE                      >> job${PBS_JOBID%%.*}.log 
echo PBS: working directory is $PBS_O_WORKDIR in $PBS_O_HOST >> job${PBS_JOBID%%.*}.log 
echo PBS: execution mode is $PBS_ENVIRONMENT                 >> job${PBS_JOBID%%.*}.log 
echo PBS: job identifier is $PBS_JOBID                       >> job${PBS_JOBID%%.*}.log 
echo PBS: job name is $PBS_JOBNAME                           >> job${PBS_JOBID%%.*}.log 
echo PBS: node file is $PBS_NODEFILE                         >> job${PBS_JOBID%%.*}.log 
echo PBS: current home directory is $PBS_O_HOME              >> job${PBS_JOBID%%.*}.log 
echo PBS: PATH = $PBS_O_PATH                                 >> job${PBS_JOBID%%.*}.log 
echo ------------------------------------------------------  >> job${PBS_JOBID%%.*}.log 
date                                                         >> job${PBS_JOBID%%.*}.log 
echo "Started"                                               >> job${PBS_JOBID%%.*}.log 


## Main part (can be modified)
echo Flocking_Confluent
echo $(which python) >> job${PBS_JOBID%%.*}.log 
## MATCH
## Insert job name below

##./a.out	2>> job${PBS_JOBID%%.*}.log	#TO EXECUTE THE JOB!
##./b.out	2>> job${PBS_JOBID%%.*}.log	#TO EXECUTE THE JOB!
##./c.out	2>> job${PBS_JOBID%%.*}.log	#TO EXECUTE THE JOB!


## DO NOT TOUCH
date                                                         >> job${PBS_JOBID%%.*}.log 
echo "Ended"                                                 >> job${PBS_JOBID%%.*}.log 
