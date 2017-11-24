#! /bin/bash

git add TP
git commit -m 'x'
git push

DIR=`pwd`
URL=`git config --get remote.origin.url`

cd ../Archives
BASE=`pwd`

git clone ${URL}
rm -rf AMS_I03/.git

cd AMS_I03/2017_2018/TP
d=`date +%Y_%m_%d_%H_%M`

dirs=`ls -d I03_TP?`
echo ${dirs}

for i in ${dirs}
do
    tar cfz ${BASE}/${i}.tgz ${i}
    cp ${BASE}/${i}.tgz ${BASE}/${i}_${d}.tgz
done

cd ${BASE}
rm -rf AMS_I03

cd ${DIR}

