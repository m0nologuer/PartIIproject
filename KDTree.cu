#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "KDTree.h"
#ifdef USE_CUDA


extern "C" __global__ void findNeighbourArray(float *positions, const int* indicies,
	const int indices_size, const float radius, int* neighbours, int offset, int scramble)
{
	//data shared between each block
	__shared__ bool neighbour_check[MAX_THREADS]; //since indices_size < MAX_THREADS

	//sort out indexes
	int id = blockDim.x * blockIdx.x + threadIdx.x + offset; //thread id, from 0 to indices_size * indices_size
	int choice_offset = id*scramble%indices_size; //so we look at a different selection each time
	int particle_index = blockDim.x * blockIdx.x / indices_size; //the particle we sum at, chosen per block
	int neighbour_index = (threadIdx.x + choice_offset) % indices_size; //the neighbour we check, per thread

	//perform comparison
	if (particle_index < indices_size)
	{
		float* pos = positions + indicies[neighbour_index]*4;
		float* starting_pos = positions + indicies[particle_index]*4;
		float distance = 0;
		for (int i = 0; i < 3; i++)
			distance += (pos[i] - starting_pos[i])*(pos[i] - starting_pos[i]);

		neighbour_check[neighbour_index] = (distance < radius) && (particle_index != neighbour_index);
		//this block should fill up information on neighbours for this one particle
 	}
	__syncthreads();

	
	//perform reduction if we get the reductive thread
	if (neighbour_index == 0)
	{
		int neighbour_written_counter = 0;
		int i = 0;
		while (neighbour_written_counter < MAX_NEIGHBOURS)
		{
			//if we find a neighbour
			if (neighbour_check[(i + choice_offset) % indices_size] && i < indices_size)
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
void KDTree::batchNeighbouringParticles(double rad, float* particles_CUDA, int* neighbours_CUDA)
{
	for (int i = 0; i < all_nodes.size(); i++)
		if (all_nodes[i]->particle_blob) // if we're at a leaf node
			batchNeighbouringParticles_CUDA(all_nodes[i], rad, particles_CUDA, neighbours_CUDA);

}
void KDTree::batchNeighbouringParticles_CUDA(const Node* n, double rad, float* particles_CUDA, int* neighbours_CUDA)
{
	float volume = 100;
	float rad_volume = 4 / 3 * rad*rad*rad; //volume of the sphere
	float p = rad_volume / volume;
	//choose n so we have 90% chance we find $MAX_NEIGHBOURS out of n draws in the volume
	//so B(n,rad_volume/volume) > $MAX_NEIGHBOURS = .90
	int trials = MAX_NEIGHBOURS / p + 1;
	while (trials < n->blob_size)
	{
		//estimate for the lower tail gives us
		float success_chance = exp((-1 / (2*p))* (pow(trials*p - MAX_NEIGHBOURS, 2) / trials));
		if (success_chance > 0.9)
			break;
		trials *= 1.5;
	}

	// Launch the CUDA Kernel
	int threadsPerBlock = max(min(n->blob_size, trials), 32); //so we can have one particle computed per block
	int blocksPerGrid = min(n->blob_size, 65536 / threadsPerBlock - 1); //avoid overstepping max no. of threads
	int offset = 0;
	
	//launch until all subsets have been considered
	while (offset < n->blob_size)
	{
		findNeighbourArray << <blocksPerGrid, threadsPerBlock >> >(particles_CUDA, n->particle_blob,
			n->blob_size, rad*rad, neighbours_CUDA, offset, rand());
		offset += blocksPerGrid;
		cudaDeviceSynchronize();

	}
	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
}
#endif