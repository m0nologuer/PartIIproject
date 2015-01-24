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

#ifdef USE_KDTREE
	//start by building the k-D tree if we don't have one
	if (!tree)
	{
		tree = KDTree::treeFromFloats(g_particle_position_data, MAX_PARTICLE_COUNT);
		RECORD_SPEED("	Build K-D tree  %d ms \n");
	}   
	
	//parrell batch processing of neighbours
	tree->batchNeighbouringParticles(h, positions_CUDA, neighbours_CUDA);
	RECORD_SPEED("	Find neighbouring particles  %d ms \n");
	//refresh kdtree periodically
	if (rand() % 5 == 0)
	{
		delete tree;
		tree = NULL;
	}

#else
	findNeighbours_CUDA(delta);
#endif

	//Perform collision detection, solving
	solverIterations_CUDA(delta);
	RECORD_SPEED("	CUDA solver iteraions  %d ms \n");

	mem_size = sizeof(float) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(lambdas_array, particle_lambdas_CUDA, mem_size, cudaMemcpyDeviceToHost));

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
extern "C" __global__ void buildGrid(float *predicted_pos, int *grid, float h,
	float min_x, float min_y, float min_z)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT

	//fill up grid 
	if (index < MAX_PARTICLE_COUNT)
	{
		float * pos = &predicted_pos[3 * index];
		int i = (pos[0] - min_x) / h;
		int j = (pos[1] - min_y) / h;
		int k = (pos[2] - min_z) / h;

		bool within_range = (i > 0 && i < h * GRID_RES) && (j > 0 && j < h * GRID_RES)
			&& (k > 0 && k < h * GRID_RES);
		if (within_range)
			grid[index] = i * GRID_RES * GRID_RES + j * GRID_RES + k;
		else
			grid[index] = -1;
	}
}
__device__ void swap_grid(int *array, int i1, int i2)
{
	int temp = array[i1];
	array[i1] = array[i2];
	array[i2] = temp;
}
__device__ void swap_grid_block(int *array, int index, int block_size, bool direction)
{
	//block and block index
	int block = index / block_size;
	int block_i = index % block_size;
	if (block_i * 2 > block_size)
	{
		block += (MAX_PARTICLE_COUNT / 2) / block_size;
		block_i = (block_size - block_i);
	}
	int i1 = block*block_size + block_i;
	int i2 = 0;
	
	if (direction)
		i2 = block*block_size + (block_size - block_i);
	else
		i2 = block*block_size + (block_i + block_size / 2);

	swap_grid(array, i1, i2);

}
extern "C" __global__ void bitonicSortGrid(int *grid)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICLE_COUNT/2
	if (!(index < MAX_PARTICLE_COUNT/2))
		return;

	for (int i = 2; i <= MAX_PARTICLE_COUNT; i *= 2)
	{
		swap_grid_block(grid, index, i, true);
		__syncthreads();

		for (int j = i/2; j > 1; j /= 2)
		{
			swap_grid_block(grid, index, j, false);
			__syncthreads();
		}
	}

}
extern "C" __global__ void sortGrid(int *grid)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT
	if (!(index < MAX_PARTICLE_COUNT))
		return;
	int grid_value = grid[index];

	//mergesort based on the roundedup value of predicted pos, then by index
	int start = index;
	int block_size = 1;
	while (start % (block_size * 2) == 0 && block_size < MAX_PARTICLE_COUNT)
	{
		int i = 0;
		int j = 0;
		int w = 0;
		int* current_block = &grid[2 * start];

		//merge sort this section
		while (i < block_size && j < block_size && j + start < MAX_PARTICLE_COUNT)
		{
			if (current_block[i] < current_block[(block_size + j)])
				swap_grid(current_block, w++, i++);
			else
				swap_grid(current_block, w++, block_size + j++);
		}

		while (i < block_size)
			swap_grid(current_block, w++, i++);

		while (j < block_size && j + block_size + start < MAX_PARTICLE_COUNT)
			swap_grid(current_block, w++, block_size + j++);


		block_size *= 2;
		__syncthreads();
	}

}
extern "C" __global__ void buildIndex(int *grid, int *grid_index)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT
	if (!(index < GRID_RES*GRID_RES*GRID_RES))
		return;

	//search for grid value, where each cell is a rounded up predicted pos value
	int start_index = MAX_PARTICLE_COUNT / 2;
	int end_index = MAX_PARTICLE_COUNT / 2;
	for (int step = start_index / 2; step > 0; step /= 2)
	{
		//find index to start of block
		if (grid[start_index] < index)
			start_index += step;
		else
			start_index -= step;

		//and end
		if (grid[end_index] < index)
			end_index -= step;
		else
			end_index += step;
	}
	if (grid[start_index] == index)
		grid_index[2*index] = max(start_index - 1, 0);
	else
		grid_index[2*index] = -1;

	if (grid[end_index] == index)
		grid_index[2 * index + 1] = end_index + 1;
	else
		grid_index[2 * index + 1] = -1;
}  
extern "C" __global__ void findNeighboursGrid(float *predicted_pos, int *grid, int* grid_index, int scramble,
	int* neighbour_indexes, float min_x, float min_y, float min_z, float h)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT
	if (!(index < MAX_PARTICLE_COUNT))
		return;

	//find cube on grid
	float * pos = &predicted_pos[3 * index];
	int i_i = (pos[0] - min_x) / h;
	int i_j = (pos[1] - min_y) / h;
	int i_k = (pos[2] - min_z) / h;

	bool within_range = (i_i > 0 && i_i < h * GRID_RES) && (i_j > 0 && i_j < h * GRID_RES)
		&& (i_k > 0 && i_k < h * GRID_RES);
	if (!within_range)
		return;

	//find neighbours
	int found_neighbours = 0;
	float h_2 = h*h;
	//loop through neighbouring squares
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
			{
				int r_i = ((i*scramble + scramble) % 3 - 1 + i_i + 1) % GRID_RES;
				int r_j = ((j*scramble + scramble) % 3 - 1 + i_j + 1) % GRID_RES;
				int r_k = ((k*scramble + scramble) % 3 - 1 + i_k + 1) % GRID_RES;

				int grid_offsetted_val = r_i * GRID_RES * GRID_RES + r_j * GRID_RES + r_k;
				int start = grid_index[2*grid_offsetted_val];
				if (start < 0)
					continue;
				int end = grid_index[2 * grid_offsetted_val + 1];
				//check all the neighbouring particles
				for (int counter = start; counter < end; counter++)
					if (found_neighbours < MAX_NEIGHBOURS)
					{
						int offset = (counter*scramble + index) % (end - start) + start;
						float distance = 0;
						for (int i = 0; i < 3; i++)
							distance += (pos[i] - predicted_pos[3 * offset + i]) * (pos[i] - predicted_pos[3 * offset + i]);

						//store the index to the location in the rpedicted pos array
						if (distance < h_2)
						{ 
							neighbour_indexes[index*MAX_NEIGHBOURS + found_neighbours] = offset;
							found_neighbours++;
						}
					}
			}
	//fill up the rest of the neighbours
	while (found_neighbours < MAX_NEIGHBOURS)
	{
		neighbour_indexes[index*MAX_NEIGHBOURS + found_neighbours] = -1;
		found_neighbours++;
	}
}
extern "C" __device__  void applyForce(float* speed , float delta)
{
	speed[0] *= 0.9;
	speed[1] += -9.81 * delta;
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

			GL_colors[4 * i] = (char)((5.0 - life[i]) * 255);

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

				life[i] = 5.0f; //lasts 5 second
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

		float* pj_speed = speed +  particle_neighbour_index * 3; //indexing by rows and columns

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
	mem_size = sizeof(int) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMalloc((void **)&grid_CUDA, mem_size));
	mem_size = sizeof(int) * GRID_RES * GRID_RES * GRID_RES * 2;
	gpuErrchk(cudaMalloc((void **)&grid_index_CUDA, mem_size));

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
	gpuErrchk(cudaFree((void **)&positions_CUDA));
	gpuErrchk(cudaFree((void **)&grid_CUDA));
	gpuErrchk(cudaFree((void **)&grid_index_CUDA));
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
	 
	for (int i = 0; i < 5; i++)
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
void ParticleContainer::findNeighbours_CUDA(float delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	//find neigbours for next iteration
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	buildGrid << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA, grid_CUDA,
		5, -25, 75, -25);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Build grid  %d ms \n");

	//sort the grid
	bitonicSortGrid << <blocksPerGrid, threadsPerBlock >> >(grid_CUDA);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Sort grid  %d ms \n");

	//build index
	blocksPerGrid = GRID_RES * GRID_RES * GRID_RES / THREADS_PER_BLOCK + 1;
	buildIndex << <blocksPerGrid, threadsPerBlock >> >(grid_CUDA, grid_index_CUDA);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Build index  %d ms \n");

	//find neighbours
	blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	findNeighboursGrid << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA, grid_CUDA, 
		grid_index_CUDA, rand(), neighbours_CUDA, -25, 75, -25, 5);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Use grid to find particles  %d ms \n");


	size_t mem_size = sizeof(int) * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(grid_array, grid_CUDA, mem_size, cudaMemcpyDeviceToHost));
	mem_size = sizeof(int) * GRID_RES * GRID_RES * GRID_RES * 2;
	gpuErrchk(cudaMemcpy(grid_index_array, grid_index_CUDA, mem_size, cudaMemcpyDeviceToHost));
	mem_size = sizeof(int) * MAX_NEIGHBOURS * MAX_PARTICLE_COUNT;
	gpuErrchk(cudaMemcpy(neighbours_array, neighbours_CUDA, mem_size, cudaMemcpyDeviceToHost));

 	RECORD_SPEED("	Find neighbouring particles  %d ms \n");
	time++;

}
#endif