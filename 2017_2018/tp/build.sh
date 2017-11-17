#! /bin/bash

for i in Util Ex*
do
   cd $i
   ./build.sh
   cd ..
done

