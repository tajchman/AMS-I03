#! /bin/bash

DIR=`pwd`

cd $DIR
mkdir -p test
cd test
rm -f r.gnp s1 s2 r

echo "
set term pdf
set output 'r.pdf'
set xlabel 'Taille de bloc'
set ylabel 'Temps de calcul transposition'
" > r.gnp

function run {

    n=$1
    echo ${code} bloc = $n
    echo $n >> s1
    ${code} 10000 10000 $n >& x
    cat x
    grep 'cpu' x | sed -e 's/cpu time     ://' -e 's/s//' >> s2
    cat s2
}

code=../build/transposee/transpose3
for i in {1..10}
do
    let n=i
    run $n
done

for i in {15..50..5}
do
    let n=i
    run $n
done

sed -i -e 's/cpu time     ://' -e 's/s//' s2
paste s1 s2 > r
echo "plot 'r' using 1:2 w lp ps 0.3 title 'transpose / bloc'" >> r.gnp

gnuplot r.gnp

