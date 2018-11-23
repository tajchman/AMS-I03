#! /bin/bash

if [ -f sinus.dat ]
then
    titres=`head -n 1 sinus.dat`
    titres=(${titres// / })
    echo ${titres[1]}
    echo "
set output 'sinus.pdf'
set term pdf
plot 'sinus.dat' using 1:2 w l lw 3 title '"${titres[2]}"', 'sinus.dat' using 1:3 w l lw 3 title '"${titres[3]}"'
" | gnuplot
    evince sinus.pdf &
fi
