#! /bin/bash

sed "s/NN/$1/" < ../job.in > ../job.sh

qsub ../job.sh
