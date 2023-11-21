Two variants of parallel quicksort: hyperquicksort and PSRS.

Compile hyperquicksort with: mpicc -g -Wall -o hqs hyperquicksort.c   

Execute hyperquicksort with: mpirun -np [number of processes MUST BE A POWER OF 2] ./hqs [input file]  [output file]

Compile PSRS with: mpicc -g -Wall -o psrs psrs.c -lm

Execute PSRS with: mpirun -np [number of processes] ./psrs [input file] [output file]



A sample input file is given. It consists of the number of elements to be sorted on the first line, followed by a newline-separated list of elements.
