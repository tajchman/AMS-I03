#! /bin/bash

rm -rf I03*
DIR=`pwd`
RECEPTION='/opt/devs/workspace/AMS_I03/2017_2018/Envoi'

cd ${RECEPTION}
ARCHIVES=`ls I03*tgz`
cd ${DIR}

for i in ${ARCHIVES}
do
    echo ${i}
    cp ${RECEPTION}/$i .
    tar xfz $i
done
