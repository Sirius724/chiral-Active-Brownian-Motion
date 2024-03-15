#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -q gpu.q@budou                                                                                     

#$ -N cABP_CUDA
#$ -o outfile/
#$ -l gpu=1
export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu

#cd /home2/kangeun/workspace/GPU_MD/Lecture12(MPS)
mpirun -n 1 cABP_CUDA.out $1 $2 $3 $4 # execute file
