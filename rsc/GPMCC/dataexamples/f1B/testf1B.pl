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
    $y = 0.0769983*pow($p[0],3)-0.484192*pow($p[0],2)+1.61142;

    for ($i = 0; $i < $#p; $i++) {
	printf("%f ", $p[$i]);
    }
    printf("%f\n", $y);
}
