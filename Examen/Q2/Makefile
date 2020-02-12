all: Q2 Q2_fine_grain Q2_coarse_grain

%: %.cxx
	g++ -fopenmp $^ -o $@

clean:
	rm -f Q2 Q2a Q2b
