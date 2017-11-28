#! /bin/bash

./build/PoissonSeq  it=10  | grep "cpu time"
./build/PoissonSeq  it=10  convection | grep "cpu time"
./build/PoissonSeq  it=10  diffusion | grep "cpu time"
./build/PoissonSeq  it=10  convection diffusion | grep "cpu time"
     
