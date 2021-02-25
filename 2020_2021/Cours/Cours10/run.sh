#! /bin/bash

for i in Poisson*
do
   cd $i
   python run.py 8 n0=800 n1=800 n2=800
   cd -
done
