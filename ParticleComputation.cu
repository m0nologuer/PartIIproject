// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "ParticleContainer.h"
#ifdef USE_CUDA  

extern "C" __device__ float W_poly6_cuda(Particle::vec3 r, float h)
{
	double radius = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius < h_squared)
	{
		//constant is 315/64pi
		double result = pow(h_squared - radius, 3.0) * 1.56668147 / h_9;
		return result;
	}
	else //ignore particles outside a certain large radius
		return 0;
}
extern "C" __device__ Particle::vec3 dW_poly6_cuda(Particle::vec3 r, float h)
{
	double radius_2 = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius_2 < h_squared)
	{
		//constant is 315/64pi
		double radius = sqrt(radius_2);
		double result = -6 * radius * pow(h_squared - radius_2, 2.0) * 1.56668147 / h_9;
		Particle::vec3 grad;
		grad.x = r.x * result;
		grad.y = r.y * result;
		grad.z = r.z * result;
		return grad;
	}
	else //ignore particles outside a certain large radius
	{
		Particle::vec3 zero;
		zero.x = 0;
		zero.y = 0;
		zero.z = 0;
		return zero;
	}
}
extern "C" __device__ Particle::vec3 dW_spiky_cuda(Particle::vec3 r, float h)
{
	double radius_2 = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius_2 < h_squared)
	{
		//constant is 15/pi
		double radius = sqrt(radius_2);
		double result = -3 * pow(h - sqrt(radius), 2.0) * 4.77464829 / h_6;
		Particle::vec3 grad;
		grad.x = r.x * result;
		grad.y = r.y * result;
		grad.z = r.z * result;
		return grad;
	}
	else //ignore particles outside a certain large radius
	{
		Particle::vec3 zero;
		zero.x = 0;
		zero.y = 0;
		zero.z = 0;
		return zero;
	}
}
extern "C" __global__ void solverIterationPositions(Particle *particles, const int *neighbour_indexes,
	float h, float Wq, float corr_k, int n)
{
	__shared__ Particle::vec3 predicted_pos_contributions[THREADS_PER_BLOCK];

	//sort out indexes
	int id = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICLE_COUNT * MAX_NEIGHBOURS
	int particle_index = id / MAX_NEIGHBOURS; //the particle we sum at
	if (particle_index > MAX_PARTICLE_COUNT)
		return;
	int storage_index = threadIdx.x;
	bool reducer = threadIdx.x % MAX_NEIGHBOURS == 0;

	//calculate contributions to positions
	Particle pi = particles[particle_index];
	if (!(pi.life > 0))
		return;

	int particle_neighbour_index = neighbour_indexes[id];
	if (particle_neighbour_index < 0) //if there's no neighbour at this location
	{
		predicted_pos_contributions[storage_index].x = 0;
		predicted_pos_contributions[storage_index].y = 0;
		predicted_pos_contributions[storage_index].z = 0;
	}
	else
	{
		Particle pj = particles[particle_neighbour_index]; //indexing by rows and columns
		
		Particle::vec3 distance;
		distance.x = (pi.predicted_pos.x - pj.predicted_pos.x);
		distance.y = (pi.predicted_pos.y - pj.predicted_pos.y);
		distance.z = (pi.predicted_pos.z - pj.predicted_pos.z);

		double s_corr = W_poly6_cuda(distance, h) / Wq;
		s_corr = -corr_k * pow(s_corr, n);

		predicted_pos_contributions[storage_index] = dW_spiky_cuda(distance, h);
		predicted_pos_contributions[storage_index].x *= (pi.lambda + pj.lambda + s_corr);
		predicted_pos_contributions[storage_index].y *= (pi.lambda + pj.lambda + s_corr);
		predicted_pos_contributions[storage_index].z *= (pi.lambda + pj.lambda + s_corr);
 	}
	__syncthreads();

	if (reducer)
	{
		float x, y,z = 0;
		for (int j = 0; j < MAX_NEIGHBOURS; j++)
		{
			x += predicted_pos_contributions[storage_index + j].x;
			y += predicted_pos_contributions[storage_index + j].y;
			z += predicted_pos_contributions[storage_index + j].z;
		}
		particles[particle_index].predicted_pos.x = x;
		particles[particle_index].predicted_pos.y = y;
		particles[particle_index].predicted_pos.z = z;

	}
}
extern "C" __global__ void solverIterationLambdas(Particle *particles, const int *neighbour_indexes,
	float p0, float h)
{
	//data shared between each block
	__shared__ float lambda_numerators[THREADS_PER_BLOCK];
	__shared__ float lambda_denominators[THREADS_PER_BLOCK];

	//sort out indexes
	int id = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICLE_COUNT * MAX_NEIGHBOURS
	int particle_index = id / MAX_NEIGHBOURS; //the particle we sum at
	int neighbour_index = id % MAX_NEIGHBOURS; //the particle we sum at
	if (particle_index > MAX_PARTICLE_COUNT)
		return;
	int storage_index = threadIdx.x;
	bool reducer = threadIdx.x % MAX_NEIGHBOURS == 0;
	
	//calculate contributions to lambda
	Particle pi = particles[particle_index];
	if (!(pi.life > 0))
		return;

	int particle_neighbour_index = neighbour_indexes[id];
	if (neighbour_index < 0) //if there's no neighbour at this location
	{
		lambda_numerators[storage_index] = 0;
		lambda_denominators[storage_index] = 0;
	}
	else
	{
		Particle pj = particles[particle_neighbour_index]; //indexing by rows and columns
		Particle::vec3 distance; 
		distance.x = (pi.predicted_pos.x - pj.predicted_pos.x);
		distance.y = (pi.predicted_pos.y - pj.predicted_pos.y);
		distance.z = (pi.predicted_pos.z - pj.predicted_pos.z);

		lambda_numerators[storage_index] = W_poly6_cuda(distance, h);

		Particle::vec3 d_distance = dW_poly6_cuda(distance, h);
		lambda_denominators[storage_index] = -d_distance.x * pj.speed.x + d_distance.y * pj.speed.y
			+ d_distance.z * pj.speed.z;
	}
	__syncthreads();
	
	//reduction
	if (reducer)
	{
		float numerator, denominator = 0;
		for (int j = 0; j < MAX_NEIGHBOURS; j++)
		{
			numerator += lambda_numerators[storage_index + j];
			denominator += lambda_denominators[storage_index + j];
		}
		numerator = numerator / p0 - 1;
		denominator = denominator / p0;
		particles[particle_index].lambda = numerator / denominator;
	}
}

void ParticleContainer::intialize_CUDA()
{
	size_t mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMalloc((void **)&container_CUDA, mem_size));

	mem_size = sizeof(int) * MAX_PARTICLE_COUNT * MAX_NEIGHBOURS;
	neighbour_array = (int*)malloc(mem_size);
	gpuErrchk(cudaMalloc((void **)&neighbours_CUDA, mem_size));
} 
void ParticleContainer::cleanup_CUDA()
{
	free(neighbour_array);
	gpuErrchk(cudaFree((void **)&container_CUDA));
	gpuErrchk(cudaFree((void **)&neighbours_CUDA));
}
void ParticleContainer::solverIterations_CUDA(void)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;
	//copy the data across
	size_t mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(container_CUDA,container, mem_size, cudaMemcpyHostToDevice));
	//mem_size = sizeof(int) * MAX_PARTICLE_COUNT * MAX_NEIGHBOURS;
	//gpuErrchk(cudaMemcpy(neighbours_CUDA,neighbour_array, mem_size, cudaMemcpyHostToDevice));

	RECORD_SPEED("		Copy data across  %d ms \n");

	//kernel properties
	int threadsPerBlock = THREADS_PER_BLOCK; //so we can have  multiple blocks on processors
	int blocksPerGrid = MAX_PARTICLE_COUNT * MAX_NEIGHBOURS / THREADS_PER_BLOCK + 1;
	 
	for (int i = 0; i < iteration_count; i++)
	{
		// Launch the CUDA Kernel for lambdas
		solverIterationLambdas << <blocksPerGrid, threadsPerBlock >> >(container_CUDA, neighbours_CUDA, p0, h);

		//Error handling
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaGetLastError());

		// Launch the CUDA Kernel for positions
		solverIterationPositions << <blocksPerGrid, threadsPerBlock >> >(container_CUDA, neighbours_CUDA, h, 
			Wq, corr_k, n);
		//Error handling
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaGetLastError());
	}

	RECORD_SPEED("		Kernel iterations  %d ms \n");

	//copy the data back
	mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(container, container_CUDA, mem_size, cudaMemcpyDeviceToHost));

	RECORD_SPEED("		Copy data back  %d ms \n");
}
#endif