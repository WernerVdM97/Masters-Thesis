#!/usr/bin/perl

sub pow {
    my ($b, $e) = @_;
    $ti = 1;
    for ($i = 0; $i < $e; $i++) {
        $ti *= $b;
    }
    return $ti;
}

while (<STDIN>) {
    @p = split(/ /, $_);
    $x = $p[1];
    $y = 0.0971895*pow($p[0],3)-0.911722*pow($p[0],2)+1.9686*pow($p[0],1)-0.193693;

    $err1 += ($x - $y)*($x - $y);
    $err2 += ($x - $z)*($x - $z);
    #for ($i = 0; $i < $#p; $i++) {
	#printf("%f ", $p[$i]);
    #}
    printf("%f %f %f\n", $p[0], $x, $y);
}
printf("%f %f\n", $err1 / 2000, $err2 / 2000);
