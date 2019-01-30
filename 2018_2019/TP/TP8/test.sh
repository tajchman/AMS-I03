#! /bin/bash

./install/eqn_cpu 200 10000 >& o1.txt
./install/eqn_cpu 200 10000 >& o2.txt
grep xxx o1.txt | sort > p1.txt
grep xxx o2.txt | sort > p2.txt
