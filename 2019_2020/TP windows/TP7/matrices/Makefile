all: mult

mult: main.cu sMatrice.h
	nvcc -g main.cu -o mult

clean:
	rm -f mult
