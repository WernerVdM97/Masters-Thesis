set terminal postscript portrait color
set xlabel "x"
set ylabel "y"
plot "f1B.out" using 1:2 title "f2" with dots, "../../gapolycurve/f1B/f1B.ga" using 1:2 title "GA" with lines, "f1B.out" using 1:3 title "NN" with lines