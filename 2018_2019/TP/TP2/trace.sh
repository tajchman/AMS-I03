#! /bin/bash

if [ -f sinus.dat ]
then  
    echo "
set output 'sinus_new.pdf'
set term pdf
plot 'sinus.dat' using 1 w l lw 3 title 'exact', 'sinus.dat' using 2 w l lw 3 title 'approché'
" | gnuplot

mv sinus_new.pdf sinus.pdf
evince sinus.pdf &
fi
