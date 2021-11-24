#!/usr/bin/perl

open FILE, "<f1A.plot";

while (<FILE>) {
    $array[$max] = $_;
    $max++;
}

close FILE;

for ($i = 0; $i < 3 * $max; $i++) {
    $a = rand($max) % $max;
    $b = rand($max) % $max;

    $val = $array[$a];
    $array[$a] = $array[$b];
    $array[$b] = $val;
}

open FILEA, ">f1A.training";
open FILEB, ">f1A.general";
open FILEC, ">f1A.test";


$i = 0;
foreach $item (@array) {
    if ($i % 12 == 0) {
	print FILEB $item;
    } elsif ($i % 12 == 1) {
	print FILEC $item;
    } else {
	print FILEA $item;
    }
    $i++;
}

close FILEA;
close FILEB;
close FILEC;
