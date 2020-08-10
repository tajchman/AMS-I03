all: ex1a.pdf

%.dvi: %.tex
	latex $<

%.tex: %.py
	python $<

%.ps: %.dvi
	dvips $<

%.pdf: %.ps
	ps2pdf $<

clean:
	\rm -f *.tex *~ *.pyc *.aux *.log *.out *.eps *.pdf
