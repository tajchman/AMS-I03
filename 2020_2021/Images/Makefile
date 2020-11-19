all: multithread0.pdf \
     multithread1.pdf \
	 multithread2.pdf \
	 multithread3.pdf \
	 multithread4.pdf \
	 multithread5.pdf \
	 sequentiel.pdf \
	 sequentiel0.pdf \
	 sequentiel1.pdf \
	 sequentiel2.pdf \
	 sequentiel3.pdf \
	 sequentiel4.pdf \
	 sequentiel5.pdf \
	 sequentiel6.pdf \
	 sequentiel7.pdf \
	 ex2_memoire_1.pdf \
	 ex2_memoire_2.pdf \
	 ex2_memoire.pdf

%.dvi: %.tex
	latex $<

multithread%.tex: multithread.py
	python $<

sequentiel%.tex: sequentiel.py
	python $<

ex2_memoire%.tex: ex2_memoire.py
	python $<
ex2_memoire.tex: ex2_memoire.py
	python $<

%.ps: %.dvi
	dvips $<

%.pdf: %.ps
	ps2pdf $<

clean:
	\rm -f *.tex *~ *.pyc *.aux *.log *.out *.eps *.pdf
