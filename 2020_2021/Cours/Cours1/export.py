#! /bin/bash

rm -rf export
mkdir -p export
cp -rf Exemples1 *pdf export

cd export
for f in build CMakeFiles CMakeCache.txt cmake_install.cmake "results*" "*.exe" .vscode "*~" Makefile
do
    find Exemples1 -name $f -prune -exec \rm -rf {} \;
done

tar cfz Exemples1.tar.gz Exemples1
zip -rq Exemples1.zip Exemples1

rm -rf Exemples1
