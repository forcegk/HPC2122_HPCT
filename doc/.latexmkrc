$latex = "latex -synctex=1 -shell-escape -halt-on-error %O %S";
$pdflatex = "pdflatex -synctex=1 -shell-escape -halt-on-error %O %S";
$pdf_previewer = '';
$sleep_time = 1;
$view = 'none';
$pdf_mode = 1;
$clean_ext .= 'acn acr alg aux bbl fdb_latexmk fls glg* glo* gls* idx ilg ' .
              'ind ist nav nlo nls nlg loc lof lot log out pyg pytxcode run.xml slo ' .
              'sls slg snm soc synctex.gz tdo thm toc upa vrb xdy _minted-%R/* _minted-%R ' .
              'pythontex-files-%R *-eps-converted-to.pdf *.gnuplot';
