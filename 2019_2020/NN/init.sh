#! /bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/data

for f in train-images-idx3-ubyte \
             train-labels-idx1-ubyte \
             t10k-images-idx3-ubyte \
             t10k-labels-idx1-ubyte
do
    if [ ! -f $DIR/data/${f} ]
    then
        cd $DIR/data
        wget http://yann.lecun.com/exdb/mnist/${f}.gz
        gunzip ${f}.gz  
    fi
done     
             
