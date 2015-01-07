// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "ParticleContainer.h"
#ifdef USE_CUDA  

void ParticleContainer::CUDAloop(double delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	//start by building the k-D tree if we don't have one
	if (!tree)
	{
		tree = new KDTree(container, max_particle_count);
	}
	RECORD_SPEED("	Build K-D tree  %d ms \n");

	//parrell batch processing of neighbours
	tree->batchNeighbouringParticles(h, container_CUDA, neighbours_CUDA);
	RECORD_SPEED("	Find neighbouring particles  %d ms \n");

	size_t mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(container, container_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//Perform collision detection, solving
	solverIterations_CUDA(delta);
	RECORD_SPEED("	CUDA solver iteraions  %d ms \n");

	mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(container, container_CUDA, mem_size, cudaMemcpyDeviceToHost));


	//apply forces, generate new particles
	updateParticles_CUDA(delta);
	RECORD_SPEED("	Update particles  %d ms \n");

	mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(container, container_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//copy to GPU buffer
	mem_size = sizeof(GLfloat) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMemcpy(g_particle_position_data, positions_CUDA, mem_size, cudaMemcpyDeviceToHost));
	mem_size = sizeof(GLubyte) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMemcpy(g_particle_color_data, colors_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//refresh kdtree periodically
	if (rand() % 5 == 0)
	{
		//copy the data back
		mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
		gpuErrchk(cudaMemcpy(container, container_CUDA, mem_size, cudaMemcpyDeviceToHost));
		RECORD_SPEED("	Copy data back  %d ms \n");

		delete tree;
		tree = NULL;
	}

}
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
extern "C" __device__  void applyForce(Particle& p, float delta)
{
	p.speed.x += 0;
	p.speed.y += - 9.81* delta;
	p.speed.z += 0;
};
extern "C" __global__ void updateParticles(float delta, Particle *particles, GLfloat* positions, GLubyte* colors,
	int scramble)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT

	if (i < MAX_PARTICLE_COUNT)
	{
		Particle& p = particles[i];

		if (p.life > 0)
		{
			p.life -= delta;

			//For render buffer
			positions[4 * i] = p.pos.x;
			positions[4 * i + 1] = p.pos.y;
			positions[4 * i + 2] = p.pos.z;
			positions[4 * i + 3] = p.size;
			colors[4 * i] = p.r;
			colors[4 * i + 1] = p.g;
			colors[4 * i + 2] = p.b;
			colors[4 * i + 3] = p.a;

			// update position and speed, apply physics force
			p.speed.x = (p.predicted_pos.x - p.pos.x ) *(1 / delta);
			p.speed.y = (p.predicted_pos.y - p.pos.y ) *(1 / delta);
			p.speed.z = (p.predicted_pos.z - p.pos.z ) *(1 / delta);
			p.pos.x = p.predicted_pos.x;
			p.pos.y = p.predicted_pos.y;
			p.pos.z = p.predicted_pos.z;

			//for next frame
			applyForce(p, delta);
			p.predicted_pos.x = p.pos.x + p.speed.x * delta;
			p.predicted_pos.y = p.pos.y + p.speed.y * delta;
			p.predicted_pos.z = p.pos.z + p.speed.z * delta;
		}
		else
		{
			// move it out of the viewing frustrum
			positions[4 * i] = -1000;
			if ((i * scramble) % 100 < 1) //one hundreth of the remaining particles are brought to life
			{

				double theta = ((i * scramble) % 628)*0.01;
				double phi = ((i * scramble * scramble) % 314)*0.01;

				p.pos.x = sin(phi)*cos(theta) * 10;
				p.pos.y = cos(phi)*cos(theta) + 50;
				p.pos.z = sin(phi)*sin(theta) * 10;

				p.speed.x = theta;
				p.speed.y = -((i * scramble) % 45 + 155);
				p.speed.z = phi;

				p.predicted_pos.x = p.pos.x + p.speed.x * delta;
				p.predicted_pos.y = p.pos.y + p.speed.y * delta;
				p.predicted_pos.z = p.pos.z + p.speed.z * delta;

				p.life = 1.0f; //lasts 1 second

				//setting misc parameters randomly for now
				p.size = (((i * scramble) % 1000) / 2000.0f + 0.1f)*0.05;
				p.angle = ((i * scramble * scramble) % 100)*0.01;
				p.weight = ((i * scramble * scramble * scramble) % 100)*0.01;

				//start color
				p.r = 83;
				p.g = 119;
				p.b = 122;
				p.a = 255;
			}
		}
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

		float multiplier = (pi.lambda + pj.lambda + s_corr);

		predicted_pos_contributions[storage_index] = dW_spiky_cuda(distance, h);
		predicted_pos_contributions[storage_index].x *= multiplier;
		predicted_pos_contributions[storage_index].y *= multiplier;
		predicted_pos_contributions[storage_index].z *= multiplier;
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
	if (particle_index > MAX_PARTICLE_COUNT)
		return;
	int storage_index = threadIdx.x;
	bool reducer = threadIdx.x % MAX_NEIGHBOURS == 0;
	
	//calculate contributions to lambda
	Particle pi = particles[particle_index];
	if (!(pi.life > 0))
		return;

	int particle_neighbour_index = neighbour_indexes[id];
	if (particle_neighbour_index < 0) //if there's no neighbour at this location
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
		if (denominator > EPSILON)
			particles[particle_index].lambda = 0;
		else
			particles[particle_index].lambda = numerator / denominator;
	}
} 

void ParticleContainer::intialize_CUDA()
{
	size_t mem_size = sizeof(Particle) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMalloc((void **)&container_CUDA, mem_size));
	gpuErrchk(cudaMemcpy(container_CUDA, container, mem_size, cudaMemcpyHostToDevice));
	mem_size = sizeof(int) * MAX_PARTICLE_COUNT * MAX_NEIGHBOURS;

	mem_size = sizeof(int) * MAX_PARTICLE_COUNT * MAX_NEIGHBOURS;
	neighbour_array = (int*)malloc(mem_size);
	gpuErrchk(cudaMalloc((void **)&neighbours_CUDA, mem_size));

	//for copying to renderbuffers
	mem_size = sizeof(GLfloat) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMalloc((void **)&positions_CUDA, mem_size));
	gpuErrchk(cudaMemset((void **)&positions_CUDA,0, mem_size));

	mem_size = sizeof(GLubyte) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMalloc((void **)&colors_CUDA, mem_size));
} 
void ParticleContainer::cleanup_CUDA()
{
	free(neighbour_array);
	gpuErrchk(cudaFree((void **)&container_CUDA));
	gpuErrchk(cudaFree((void **)&neighbours_CUDA));
	gpuErrchk(cudaFree((void **)&colors_CUDA));
	gpuErrchk(cudaFree((void **)&positions_CUDA));
}
void ParticleContainer::updateParticles_CUDA(double delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;
	//update particles for next iteration
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	updateParticles << <blocksPerGrid, threadsPerBlock >> >(delta, container_CUDA,
		positions_CUDA, colors_CUDA, rand());

	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
}
void ParticleContainer::solverIterations_CUDA(float delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

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
}
#endif