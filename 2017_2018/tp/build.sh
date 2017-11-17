#! /bin/bash

for i in Utils Ex*
do
   cd $i
   ./build.sh
   cd ..
done

