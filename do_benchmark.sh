#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cola-corta
#SBATCH -c 17
#SBATCH -t 05:30:00

EXECUTABLE="./target/intel/release.out"
PROBLEM_SIZE="8192"

cd "/home/ulc/cursos/curso350/code/HPC2122_HPCT"

source env.sh

OLEVEL="-O2"
make -f Makefile_intel cleanall
make -f Makefile_intel BUILD=release OMP=1 OLEVEL=${OLEVEL}

for i in {1,2,4,8,16}; do
    echo "Executing for $i cores. OLEVEL=${OLEVEL}"
    OMP_NUM_THREADS=$i ./$EXECUTABLE ${PROBLEM_SIZE}
    echo ""
    echo ""
done

OLEVEL="-O3"
make -f Makefile_intel cleanall
make -f Makefile_intel BUILD=release OMP=1 OLEVEL=${OLEVEL}

for i in {1,2,4,8,16}; do
    echo "Executing for $i cores. OLEVEL=${OLEVEL}"
    OMP_NUM_THREADS=$i ./$EXECUTABLE ${PROBLEM_SIZE}
    echo ""
    echo ""
done
