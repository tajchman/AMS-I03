rm -f r.gnp r1.gnp r2.gnp

echo "
set term pdf
set output 'r.pdf'
" > r.gnp

function run {
    rm -f r s && touch r s
    c=$1
    code=./matmul${c}
    for i in {1..20}
    do
        let n=i*200
        echo $code $n
        echo $n >> s
        perf stat -e cache-misses ${code} $n $n >& x
        cat x >> r
    done
    grep cache-misses r > r1 && \
        sed -i -e 's/cache-misses//' -e 's/[^0-9]*//g' r1
    grep "seconds time elapsed" r > r2 && \
        sed -i -e 's/seconds time elapsed//' -e 's/,/./' -e 's/[^.0-9]*//g' r2
    paste s r1 r2 > r_${c}
    echo -n "'r_"${c}"' using 1:2 w lp title 'parcours "${c}"', " >> r1.gnp
    echo -n "'r_"${c}"' using 1:3 w lp title 'parcours "${c}"', " >> r2.gnp
}

function runb {
    rm -f r s && touch r s
    code=./matmul3
    p=$1
    for i in {1..20}
    do
        let n=i*200
        echo matmul3 $n $p
        echo $n >> s
        perf stat -e cache-misses ${code} $n $n $p >& x
        cat x >> r
    done
    grep cache-misses r > r1 && \
        sed -i -e 's/cache-misses//' -e 's/[^0-9]*//g' r1
    grep "seconds time elapsed" r > r2 && \
        sed -i -e 's/seconds time elapsed//' -e 's/,/./' -e 's/[^.0-9]*//g' r2
    paste s r1 r2 > r_3_${p}
    echo -n "'r_3_"${p}"' using 1:2 w lp title 'bloc "${p}"', " >> r1.gnp
    echo -n "'r_3_"${p}"' using 1:3 w lp title 'bloc "${p}"', " >> r2.gnp
}

echo -n "plot " > r1.gnp
echo -n "plot " > r2.gnp

run 1
run 2
for p in 50 40 30 20 10
do
    runb $p
done
echo >> r1.gnp
echo >> r2.gnp
cat r1.gnp r2.gnp >> r.gnp

gnuplot r.gnp

