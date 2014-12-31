#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "KDTree.h"
#include "GPUHelpers.h"

extern "C" __global__ void findNeighbours(const Particle *positions, const int positions_size,
	const Particle starting_pos, const double radius, bool* neighbours)
{
	int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_index < positions_size)
	{
		Particle pos = positions[thread_index];
		double distance = 0;
		distance += (pos.pos.x - starting_pos.pos.x)* (pos.pos.x - starting_pos.pos.x);
		distance += (pos.pos.y - starting_pos.pos.y)* (pos.pos.y - starting_pos.pos.y);
		distance += (pos.pos.z - starting_pos.pos.z)* (pos.pos.z - starting_pos.pos.z);

		neighbours[thread_index] = pos.life > 0 && distance < radius;
	} 
}
void KDTree::initialize_CUDA()
{
	//create output grid
	size_t size = particle_blob_size * sizeof(bool);
	output_grid = (bool*)malloc(size);
	gpuErrchk(cudaMalloc((void **)&output_grid_CUDA, size));
	gpuErrchk(cudaMemcpy(output_grid_CUDA, output_grid, size, cudaMemcpyHostToDevice));
}
//allocate memory CUDA  
void KDTree::initialize_CUDA_node(Node* n)
{
  	//create CUDA memory
	Particle *particles_CUDA = NULL;
	size_t size = n->blob_size * sizeof(Particle);

	//copy to device
	gpuErrchk(cudaMalloc((void **)&particles_CUDA, size));
	gpuErrchk(cudaMemcpy(particles_CUDA, n->particle_blob, size, cudaMemcpyHostToDevice));

	//deallocate original, set pointer to CUDA data
	delete[] n->particle_blob;
	n->particle_blob = particles_CUDA;
}
void KDTree::findNeighbouringParticles_CUDA(Node* n, Particle p,
	std::vector<Particle*>& list, double rad)
{
	cudaDeviceSynchronize();
	// Launch the CUDA Kernel
	int threadsPerBlock = 256; //so we can have  multiple blocks on processors
	int blocksPerGrid = thread_count / 256 + 1;

	findNeighbours << <blocksPerGrid, threadsPerBlock >> >(n->particle_blob, n->blob_size, p, rad, output_grid_CUDA);

	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());

	//Copy the first MAX_NEIGHBOURS neighbours into the vector
	size_t size = n->blob_size * sizeof(bool);
	gpuErrchk(cudaMemcpy(output_grid, output_grid_CUDA, size, cudaMemcpyDeviceToHost));

	int neighbour_count = 0;
	for (int i = 0; i < n->blob_size; i++)
		if (output_grid[i] && neighbour_count < MAX_NEIGHBOURS)
		{
			list.push_back(&n->particle_blob[i]);
			neighbour_count++;
		}
}
