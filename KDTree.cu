#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "KDTree.h"
#ifdef USE_CUDA


extern "C" __global__ void findNeighbourArray(const Particle *positions, const int* indicies,
	const int indices_size, const double radius, int* neighbours, int offset)
{
	//data shared between each block
	__shared__ bool neighbour_check[MAX_THREADS * 2]; //since indices_size < MAX_THREADS

	//sort out indexes
	int id = blockDim.x * blockIdx.x + threadIdx.x + offset; //thread id, from 0 to MAX_PARTICLE_COUNT * MAX_NEIGHBOURS
	int particle_index = id / indices_size; //the particle we sum at
	int neighbour_index = id % indices_size; //the neighbour we sum at

	//perform comparison
	if (particle_index < indices_size)
	{
		Particle pos = positions[indicies[neighbour_index]];
		Particle starting_pos = positions[indicies[particle_index]];
		double distance = 0;
		distance += (pos.pos.x - starting_pos.pos.x)* (pos.pos.x - starting_pos.pos.x);
		distance += (pos.pos.y - starting_pos.pos.y)* (pos.pos.y - starting_pos.pos.y);
		distance += (pos.pos.z - starting_pos.pos.z)* (pos.pos.z - starting_pos.pos.z);

		neighbours[neighbour_index] = (distance < radius) && (particle_index != neighbour_index);
	}
	__syncthreads();

	//perform reduction
	if (neighbour_index == 0)
	{
		int neighbour_written_counter = 0;
		int i = 0;
		while (neighbour_written_counter < MAX_NEIGHBOURS)
		{
			//if we find a neighbour
			if (neighbour_check[i] && i < indices_size)
			{
				neighbours[particle_index*MAX_NEIGHBOURS + neighbour_written_counter] = indicies[i];
				neighbour_written_counter++;
			}
			i++;
			//filing up blank space
			if (!(i < indices_size))
			{
				neighbours[particle_index*MAX_NEIGHBOURS + neighbour_written_counter] = -1;
				neighbour_written_counter++;
			}
		}
	}
}

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

}
//allocate memory CUDA  
void KDTree::initialize_CUDA_node(Node* n)
{
  	//create CUDA memory
	int *particles_CUDA = NULL;
	size_t size = n->blob_size * sizeof(int);

	//copy to device
	gpuErrchk(cudaMalloc((void **)&particles_CUDA, size));
	gpuErrchk(cudaMemcpy(particles_CUDA, n->particle_blob, size, cudaMemcpyHostToDevice));

	//deallocate original, set pointer to CUDA data
	delete[] n->particle_blob;
	n->particle_blob = particles_CUDA;
}
void KDTree::batchNeighbouringParticles(double rad, const Particle* particles_CUDA, int* neighbours_CUDA)
{
	for (int i = 0; i < all_nodes.size(); i++)
		if (all_nodes[i]->particle_blob) // if we're at a leaf node
			batchNeighbouringParticles_CUDA(all_nodes[i], rad, particles_CUDA, neighbours_CUDA);

}
void KDTree::batchNeighbouringParticles_CUDA(const Node* n, double rad, const Particle* particles_CUDA, int* neighbours_CUDA)
{
	// Launch the CUDA Kernel
	int threadsPerBlock = n->blob_size; //so we can have one particle computed per block
	int blocksPerGrid = min(n->blob_size, 65536 / n->blob_size)-1; //avoid overstepping max no. of threads
	int offset = 0;
	
	//launch until all subsets have been considered
	while (offset < n->blob_size)
	{
		findNeighbourArray << <blocksPerGrid, blocksPerGrid >> >(particles_CUDA, n->particle_blob,
			n->blob_size, rad, neighbours_CUDA, offset);
		offset += blocksPerGrid;
		cudaDeviceSynchronize();

	}
	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
}
#endif