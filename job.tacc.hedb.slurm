#!/bin/bash
#SBATCH -t 4:00:00     # walltime,
#SBATCH -J prad1
#SBATCH -N 8           # number of cluster nodes
#SBATCH -c 18
#SBATCH --qos=interactive
#SBATCH --ntasks-per-node=2
#SBATCH -o proton-%j.out # pathname of stdout file
#SBATCH -e proton-%j.err # pathname of stderr file, using job and first node values

source ~/apps/intelpython/module.sh
export OMP_NUM_THREADS=18
##mkdir img_coaxis
##srun -u python test_HEDB1.py -i /net/scratch4/yclu/HEDB19/coaxis/omega2018_hdf5_plt_cnt -o img_coaxis/img -id 0 140 5 --coaxis --left
##mkdir img_slotsa
##srun -u python test_HEDB1.py -i /net/scratch4/yclu/HEDB19/slotsa/omega2017_hdf5_plt_cnt -o img_slotsa/img -id 40 81 5 --coaxis --left
##mkdir img_marblevc
##srun -u python test_HEDB1.py -i /net/scratch4/yclu/Marbel17/Blue_H_ed/omega2017_hdf5_plt_cnt -o img_marblevc/img -id 40 171 5 --marblevc --left

