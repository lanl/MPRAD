#!/bin/bash
#SBATCH -t 4:00:00     # walltime,
#SBATCH -J qrad2
#SBATCH -N 8           # number of cluster nodes
#SBATCH -c 18
##SBATCH --qos=interactive
#SBATCH --ntasks-per-node=2
#SBATCH -o proton-%j.out # pathname of stdout file
#SBATCH -e proton-%j.err # pathname of stderr file, using job and first node values

source ~/apps/intelpython/module.sh
export OMP_NUM_THREADS=18
mkdir img_slotsa
srun -u python test_HEDB1.py --species_table slots --shot_table HEDBDAY0 --file_table newres --ener_table pack1 --output_base img_slotsa/img --index 50 90 5 --optaL --optb20 --optc1 --optd0 --optd1 --optd2

