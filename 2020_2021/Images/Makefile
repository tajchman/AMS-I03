all: multithread0.pdf multithread1.pdf

%.dvi: %.tex
	latex $<

multithread%.tex: multithread.py
	python $<
	mkdir -p temp
	cp multithread*.tex temp

%.ps: %.dvi
	dvips $<

%.pdf: %.ps
	ps2pdf $<

clean:
	\rm -f *.tex *~ *.pyc *.aux *.log *.out *.eps *.pdf
