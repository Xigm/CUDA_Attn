main: softmax.c attention.c main.c
	nvcc attention.c softmax.c main.c -o main 