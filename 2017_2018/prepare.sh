#! /bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

git add TP
git commit -m 'x'
git push

URL=`git config --get remote.origin.url`

ENVOI=${DIR}/Envoi
/bin/rm -rf ${ENVOI}
mkdir -p ${ENVOI}
cd ${ENVOI}
echo "ENVOI : " ${ENVOI}

git clone ${URL}
rm -rf AMS_I03/.git

cd AMS_I03/2017_2018/TP
d=`date +%Y_%m_%d_%H_%M`

dirs=`ls -d I03_TP?`

for i in ${dirs}
do
    rm -f *.tex
    tar cfz ${BASE}/${i}.tgz ${i}
    cp ${BASE}/${i}.tgz ${BASE}/${i}_${d}.tgz
done

cd ${BASE}
rm -rf AMS_I03

cd ${DIR}

