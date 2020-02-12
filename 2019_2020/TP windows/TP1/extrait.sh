#! /bin/bash

DIR=`pwd`
BASE=`basename ${DIR}`

cd ..

rm -rf temp
mkdir -p temp/${BASE}

cd temp/${BASE}
cp -rf ${DIR}/src .
cp ${DIR}/*.sh .
rm -f extrait.sh
cd ..
tar cfz ../${BASE}.tar.gz ${BASE}
cd ..
\rm -rf temp



