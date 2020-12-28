all: somme somme2 paral tasks1 tasks2

somme: somme.c
	${CC} $^ -o $@

somme2: somme2.c
	${CC} $^ -o $@

paral: paral.c
	${CC} -fopenmp $^ -o $@

tasks1: tasks1.c
	${CC} -fopenmp $^ -o $@

tasks2: tasks2.c
	${CC} -fopenmp $^ -o $@

clean:
	/bin/rm -f somme somme2 paral tasks1 tasks2

