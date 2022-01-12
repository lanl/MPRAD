#!/bin/bash

#SBATCH -J proton           # Job name
#SBATCH -o proton.o%j       # Name of stdout output file
#SBATCH -e proton.e%j       # Name of stderr error file
###SBATCH -p skx-dev       # Queue name
#SBATCH -p skx-normal           # Queue name
#SBATCH -N 8                # Total # of nodes
###SBATCH -N 4                # Total # of nodes
#SBATCH --ntasks-per-node=48
###SBATCH -t 02:00:00         # Run time (hh:mm:ss)
###SBATCH -t 00:30:00         # Run time (hh:mm:ss)
#SBATCH -t 08:00:00         # Run time (hh:mm:ss)

# Other commands must follow all #SBATCH directives...

export IBRUN_TASKS_PER_NODE=4
export OMP_NUM_THREADS=12
module load phdf5                            
#ibrun taff_affinity python test.py
##ibrun taff_affinity python test_coaxial.py
ibrun taff_affinity python test_shear3.py

##ibrun taff_affinity python test_sinp.py
