#!usr/bin/env bash

mkdir -p job_logs
mkdir -p job_logs/html_files

mv -t job_logs/ *.o[0-9]* job[0-9]*

mv -t job_logs/html_files/ *.html

rm -rf build/ 

rm *.so
