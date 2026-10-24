for file in *
do
	if [ -f $file ] ; then
		# name without extension (escape the dot because otherwise it is the "local_dir" character
		name=${file%\.*}
		echo ${name}
		cp "$file" "$name".pyx
	fi ;
done
