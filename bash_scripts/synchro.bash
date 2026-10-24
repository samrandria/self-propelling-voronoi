#!/usr/bin/env bash

# Synchronize the current python and bash scripts with the remote server tsutsuji
echo "Synchronizing the python files, please input your password."
rsync -t *.py samuel@133.11.163.28:~/synced_files/python_files
echo "Synchronizing the bash files, please input your password."
rsync -t *.bash samuel@133.11.163.28:~/synced_files/bash_files
# echo "Synchronizing the cython files, please input your password."
# rsync -rt cython_files/ samuel@133.11.163.28:~/synced_files/cython_files
