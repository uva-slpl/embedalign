#!/bin/bash
#SBATCH -p staging
#SBATCH -t 1-00:00:00
#SBATCH -J archive

jobid=$1
workspace=$2
if [[ -z $3 ]]; then 
	stem=output	
else
	stem=$3
fi
# archive the output files
tar cvf /archive/${USER}/${stem}_${jobid}.tar -C `dirname ${workspace}` `basename ${workspace}`

