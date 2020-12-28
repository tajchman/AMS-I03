#! /bin/bash

N=$1
J=$2
M=$(($N * 16))

echo $M
sed -e "s/NN/$N/" -e "s/JJ/$J/" -e "s/MM/$M/"< ./job.in > ./job.sh

qsub ./job.sh
