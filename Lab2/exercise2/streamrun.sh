#!/bin/bash

for i in {1,2,4,8,12,16,20,24,28,32};
do
	export OMP_NUM_THREADS=$i;
	echo $OMP_NUM_THREADS;
	for j in {1..5};
	do
		./a.out >> "stream$i.txt";
	done
done

