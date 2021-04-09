add_cus_dep('glo', 'gls', 0, 'makeglossaries');
add_cus_dep('acn', 'acr', 0, 'makeglossaries');
sub makeglossaries {
    if ($silent) {
        system("makeglossaries -q \"$_[0]\"");
    } else{
        system("makeglossaries \"$_[0]\"");
    }
}
push @generated_exts, 'glo', 'gls', 'glg';
push @generated_exts, 'acn', 'acr', 'alg';

$clean_ext .= ' %R.ist %R.xdy %R.loapp %R.tdo %R.auxlock %R.bbl %R.run.xml %R.xdv';
$pdf_mode = 5;        # use xelatex
$force_mode = 1;

push @extra_xelatex_options, '-synctex=1';
push @extra_xelatex_options, '--shell-escape';
