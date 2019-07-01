#! /bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

git add ${DIR}/TP
git commit -m 'x' -q
git push -q

URL=`git config --get remote.origin.url`

ENVOI=${DIR}/Envoi
/bin/rm -rf ${ENVOI}
mkdir -p ${ENVOI}
cd ${ENVOI}
echo "ENVOI : " ${ENVOI}

git clone -q ${URL}
rm -rf AMS_I03/.git

cd AMS_I03/2017_2018/TP

dirs=`ls -d I03_TP?`
echo "TP    : " ${dirs}

for i in ${dirs}
do
    rm -f ${i}/*.tex
    tar cfz ${ENVOI}/${i}.tgz ${i}
done

cd ${ENVOI}
rm -rf AMS_I03

cd ${DIR}

