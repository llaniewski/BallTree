all: main

main.o: main.cu BallTree.hpp
	nvcc -c -o $@ $<

main: main.o
	nvcc -o $@ $^
