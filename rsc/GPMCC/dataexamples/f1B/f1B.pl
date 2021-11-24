#!/usr/bin/perl

$asd = 0;
open PLOT, ">f1A.plot";

for ($t = 0; $t < 12000; $t++) {
    $x[0] = $t / 12000 * 6.28;
    #$x[0] = $t / 12000 * 1.5;
    $y[0] = sin($x[0]);
    #$y[0] = $x[0]*exp($x[0]);
    $q = $x[0];
    #$w = $y[0];
    $w = $y[0] + rand(2) - 1;
    
    $asd += ($w - $y[0])*($w - $y[0]);
    print PLOT $q." ".$w."\n";
}

close PLOT;
$asd /= 2 * ($t / 12);
printf("%f\n", $asd);
