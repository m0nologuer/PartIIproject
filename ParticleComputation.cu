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

	//apply forces, generate new particles
	updateParticles_CUDA(delta);
	RECORD_SPEED("	Update particles  %d ms \n");

	//copy to GPU buffer
	size_t mem_size = sizeof(GLfloat) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMemcpy(g_particle_position_data, positions_CUDA, mem_size, cudaMemcpyDeviceToHost));
	mem_size = sizeof(GLubyte) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMemcpy(g_particle_color_data, colors_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//start by building the k-D tree if we don't have one
	if (!tree)
	{
		tree = KDTree::treeFromFloats(g_particle_position_data, MAX_PARTICLE_COUNT);
		RECORD_SPEED("	Build K-D tree  %d ms \n");
	}
	
	//parrell batch processing of neighbours
	tree->batchNeighbouringParticles(h, positions_CUDA, neighbours_CUDA);
	RECORD_SPEED("	Find neighbouring particles  %d ms \n");

	mem_size = sizeof(int) * MAX_PARTICLE_COUNT * MAX_NEIGHBOURS;
	gpuErrchk(cudaMemcpy(neighbours_array, neighbours_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//Perform collision detection, solving
	solverIterations_CUDA(delta);
	RECORD_SPEED("	CUDA solver iteraions  %d ms \n");

	mem_size = sizeof(float) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(lambdas_array, particle_lambdas_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//refresh kdtree periodically
	if (rand() % 5 == 0)
	{
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
extern "C" __device__  void applyForce(float* speed , float delta)
{
	speed[0] += 0;
	speed[1] += -9.81* delta;
	speed[2] += 0;
};

extern "C" __global__ void updateParticles(float delta, float *pos, float* predicted_pos, float* speed,
	float* life, GLfloat* GL_positions, GLubyte* GL_colors, int scramble)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT

	if (i < MAX_PARTICLE_COUNT)
	{
	
		if (life[i]> 0)
		{
			life[i] -= delta;

			//For render buffer
			GL_positions[4 * i] = pos[3 * i];
			GL_positions[4 * i + 1] = pos[3 * i + 1];
			GL_positions[4 * i + 2] = pos[3 * i + 2];
			GL_positions[4 * i + 3] = 0.02;
			GL_colors[4 * i] = 125;
			GL_colors[4 * i + 1] = 84;
			GL_colors[4 * i + 2] = 84;
			GL_colors[4 * i + 3] = 255;

			// update position and speed, apply physics force
			for (int j = 0; j < 3; j++)
			{
				speed[3 * i + j] = (predicted_pos[3 * i + j] - pos[3 * i + j])*(1 / delta);
				pos[3 * i + j] = predicted_pos[3 * i + j];
			}
			//for next frame
			applyForce(&speed[3 * i], delta);
			for (int j = 0; j < 3; j++)
				predicted_pos[3 * i + j] = pos[3 * i + j] + speed[3 * i + j] * delta;

			//kill particles out of the viewing frustrum
			if (abs(pos[3 * i]) > 200 || abs(pos[3 * i + 1]) > 200)
				life[i] = -1;
		}
		else
		{
			// move it out of the viewing frustrum
			GL_positions[4 * i] = -1000;
			if ((i * scramble) % 103 < 20) //one half of the remaining particles are brought to life
			{

				double theta = ((i * scramble) % 628)*0.01;
				double phi = ((i * scramble * scramble) % 314)*0.01;

				pos[3 * i] = sin(phi)*cos(theta) * 10;
				pos[3 * i + 1] = cos(phi)*cos(theta) + 50;
				pos[3 * i + 2] = sin(phi)*sin(theta) * 10;

				speed[3 * i] = theta;
				speed[3 * i + 1] = -((i * scramble) % 45 + 155);
				speed[3 * i + 2] = phi;

				for (int j = 0; j < 3; j++)
					predicted_pos[3 * i + j] = pos[3 * i + j] + speed[3 * i + j] * delta;

				life[i] = 5.0f; //lasts 1 second
			}
		}
	}
} 
extern "C" __global__ void solverIterationPositions(float* predicted_pos, float* life, float* lambdas,
	const int *neighbour_indexes, float h, float Wq, float corr_k,float p0, int n)
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
	if (!(life[particle_index] > 0))
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
		float* pi = predicted_pos + particle_index * 3; //indexing by rows and columns
		float* pj = predicted_pos + particle_neighbour_index * 3; //indexing by rows and columns
		Particle::vec3 distance;
		distance.x = pi[0] - pj[0];
		distance.y = pi[1] - pj[1];
		distance.z = pi[2] - pj[2];

		
		double s_corr = W_poly6_cuda(distance, h) / Wq;
		s_corr = -corr_k * pow(s_corr, n);

		float multiplier = (lambdas[particle_index] + lambdas[particle_neighbour_index] + s_corr) / p0;

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
		predicted_pos[3 * particle_index] += x;
		predicted_pos[3 * particle_index + 1] += y;
		predicted_pos[3 * particle_index + 2] += z;

	}
}
extern "C" __global__ void solverIterationLambdas(float *predicted_pos, const float* life, 
	 float* speed, float* lambda, const int *neighbour_indexes, float p0, float h, float e0)
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
	if (!(life[particle_index] > 0))
		return;

	int particle_neighbour_index = neighbour_indexes[id];
	if (particle_neighbour_index < 0) //if there's no neighbour at this location
	{
		lambda_numerators[storage_index] = 0;
		lambda_denominators[storage_index] = 0;
	}
	else
	{
		float* pi = predicted_pos + particle_index * 3; //indexing by rows and columns
		float* pj = predicted_pos + particle_neighbour_index * 3; //indexing by rows and columns
		Particle::vec3 distance; 
		distance.x = pi[0] - pj[0];
		distance.y = pi[1] - pj[1];
		distance.z = pi[2] - pj[2];

		lambda_numerators[storage_index] = W_poly6_cuda(distance, h);

		float* pj_speed = speed + particle_neighbour_index * 3; //indexing by rows and columns

		Particle::vec3 d_distance = dW_poly6_cuda(distance, h);
		lambda_denominators[storage_index] = -(d_distance.x * pj_speed[0] + d_distance.y * pj_speed[1]
			+ d_distance.z * pj_speed[2]);
	}
	__syncthreads();
	
	//reduction
	if (reducer)
	{		
		float numerator = 0;
		float denominator = e0;
		for (int j = 0; j < MAX_NEIGHBOURS; j++)
		{
			numerator += lambda_numerators[storage_index + j];
			denominator += lambda_denominators[storage_index + j] * lambda_denominators[storage_index + j];
		}
		numerator = numerator / p0 - 1;
		denominator = denominator / p0;
		lambda[particle_index] = -numerator/denominator;
	}
} 

void ParticleContainer::intialize_CUDA()
{
	//for solvers
	size_t mem_size = sizeof(float) * MAX_PARTICLE_COUNT * 3;
	gpuErrchk(cudaMalloc((void **)&particle_positions_CUDA, mem_size));
	gpuErrchk(cudaMalloc((void **)&particle_speeds_CUDA, mem_size));
	gpuErrchk(cudaMalloc((void **)&particle_predicted_pos_CUDA, mem_size));
	
	//one float per particle
	mem_size = sizeof(float) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMalloc((void **)&particle_lambdas_CUDA, mem_size));
	gpuErrchk(cudaMalloc((void **)&particle_life_CUDA, mem_size));
	gpuErrchk(cudaMemset((void*)particle_life_CUDA,0, mem_size));

	//for neighbours
	mem_size = sizeof(int) * MAX_PARTICLE_COUNT * MAX_NEIGHBOURS;
	gpuErrchk(cudaMalloc((void **)&neighbours_CUDA, mem_size));

	//for copying to renderbuffers
	mem_size = sizeof(GLfloat) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMalloc((void **)&positions_CUDA, mem_size));
	gpuErrchk(cudaMemset((void *)positions_CUDA,0, mem_size));

	mem_size = sizeof(GLubyte) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMalloc((void **)&colors_CUDA, mem_size));

	//init data
	updateParticles << < 1, 512 >> >(0.2, particle_positions_CUDA,
		particle_predicted_pos_CUDA, particle_speeds_CUDA, particle_life_CUDA,
		positions_CUDA, colors_CUDA, 0);
	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
} 
void ParticleContainer::cleanup_CUDA()
{
	gpuErrchk(cudaFree((void **)&particle_positions_CUDA));
	gpuErrchk(cudaFree((void **)&particle_lambdas_CUDA));
	gpuErrchk(cudaFree((void **)&particle_speeds_CUDA));
	gpuErrchk(cudaFree((void **)&particle_predicted_pos_CUDA));
	gpuErrchk(cudaFree((void **)&neighbours_CUDA));
	gpuErrchk(cudaFree((void **)&colors_CUDA));
	gpuErrchk( cudaFree((void **)&positions_CUDA));
}
void ParticleContainer::updateParticles_CUDA(double delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	//update particles for next iteration
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	updateParticles << <blocksPerGrid, threadsPerBlock >> >(delta, particle_positions_CUDA, 
		particle_predicted_pos_CUDA, particle_speeds_CUDA, particle_life_CUDA,
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
	 
	for (int i = 0; i < 10; i++)
	{
		// Launch the CUDA Kernel for lambdas
		solverIterationLambdas << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA,
			particle_life_CUDA, particle_speeds_CUDA, particle_lambdas_CUDA, neighbours_CUDA, 0.5, 5, 0.1);

		//Error handling
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaGetLastError());

		// Launch the CUDA Kernel for positions
		solverIterationPositions << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA,
			particle_life_CUDA, particle_lambdas_CUDA, neighbours_CUDA, 5, 0.0121611953, 0.1, 0.5, 4);
		//Error handling
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaGetLastError());
	}
}
#endif