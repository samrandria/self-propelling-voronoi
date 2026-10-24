#!/usr/bin/env bash

targetdir=$1
echo "$targetdir"

for dir in "$targetdir"/*/*/*/logdirectory
do
	cp -t "$dir" *py
done

for dir in "$targetdir"/*/*/*/
do 
	cp -t "$dir" create_gif.py
done
