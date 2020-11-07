all: multithread0.pdf multithread1.pdf multithread2.pdf multithread3.pdf multithread4.pdf multithread5.pdf

%.dvi: %.tex
	latex $<

multithread%.tex: multithread.py
	python $<

%.ps: %.dvi
	dvips $<

%.pdf: %.ps
	ps2pdf $<

clean:
	\rm -f *.tex *~ *.pyc *.aux *.log *.out *.eps *.pdf
