#! /bin/bash

rm -rf export
mkdir -p export
cp -rf Exemples2 *pdf export

cd export
for f in build CMakeFiles CMakeCache.txt cmake_install.cmake "results*" "*.exe" .vscode "*~" Makefile
do
    find Exemples2 -name $f -prune -exec \rm -rf {} \;
done

tar cfz Exemples2.tar.gz Exemples2
zip -rq Exemples2.zip Exemples2

rm -rf Exemples2
