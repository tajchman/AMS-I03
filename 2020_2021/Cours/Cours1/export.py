#! /bin/bash

rm -rf export
mkdir -p export
cp -rf Exemples *pdf export

cd export
rm -rf */*/build  */*/results*

tar cfz Exemples.tar.gz Exemples
zip -r Exemples.zip Exemples

rm -rf Exemples
