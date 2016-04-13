#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


void transfer_board(GoBoard& input, GoBoard& output){
	GoBoard* device_input;
	GoBoard* device_output;
	cudaMalloc((void **)&device_output, sizeof(*input));
	cudaMalloc((void **)&device_input, sizeof(*input));
	cudaMemcpy(device_input, &input, sizeof(*input), 
               cudaMemcpyHostToDevice);
	
}

__global__ void run_simulation(TreeNode* node){

}
