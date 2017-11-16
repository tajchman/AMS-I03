#! /bin/bash

for i in Ex*
do
   cd $i
   ./build.sh
   cd ..
done

