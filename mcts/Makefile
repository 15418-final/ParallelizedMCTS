all:
	nvcc -O3 -arch=sm_20 --device-c CudaGo.cu mcts_cuda.cu
	nvcc -O3 -arch=sm_20 --device-link CudaGo.o mcts_cuda.o --output-file link.o
	g++ main.cpp CudaGo.o mcts_cuda.o link.o -I/usr/local/cuda/include -L/usr/local/cuda/lib/ -lcudart
clean:
	rm -rf *.o *.out
