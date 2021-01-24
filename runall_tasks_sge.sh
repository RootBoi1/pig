#!/bin/bash
#$ -cwd
#$ -l h_vmem=8G
# #$ -pe smp 4
#$ -R y
# -o and -e need to different for each user.
#$ -o logs/pig_o/$JOB_ID.o_$TASK_ID
#$ -e logs/pig_e/$JOB_ID.e_$TASK_ID

python pig.py --calc $SGE_TASK_ID
