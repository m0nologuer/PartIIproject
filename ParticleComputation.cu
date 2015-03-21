// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda.h>
#include <stdio.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include "ParticleContainer.h"

//force texture
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> external_force;

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
#ifdef ATOMIC_METHOD
	//findNeighboursAtomic_CUDA(delta);
#else
	findNeighbours_CUDA(delta);
#endif
#endif

	//Perform collision detection, solving
//	solverIterations_CUDA(delta);
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
extern "C" __device__ bool allocate_to_block(int grid_sq, int index, int *grid, int scramble,int& memory_block)
{
	int next_block = memory_block;
	//follow the chain of allocated memor  y blocks
	//use atomicCAS for atomic reads
	do{
		memory_block = next_block;
		next_block = atomicCAS(&grid[memory_block * BLOCK_SIZE + BLOCK_SIZE - 1], 0, 0); //no effect, just read
	} while (next_block != 0);

	//put the index into this block
	for (int block_id = 1; block_id < BLOCK_SIZE - 1; block_id++)
		if (atomicCAS(&grid[memory_block * BLOCK_SIZE + block_id], 0, index + 1) == 0) //atomically update index if its free
			return true;
	
	return false;
}
#define NEXT_BLOCK_INDEX(x)  x * BLOCK_SIZE 
#define BLOCK_FREE_SPACE(x)  x * BLOCK_SIZE 
#define BLOCK(x,y)  x * BLOCK_SIZE + y
extern "C" __device__ int find_free_block(int index, int *grid, int scramble)
{
	//random starting position
	int start_block = ((scramble + 1) * index * (SPARE_MEMORY_SIZE - 1) + scramble) % SPARE_MEMORY_SIZE;
	for (int i = 0; i < SPARE_MEMORY_SIZE; i++)
	{
		int current_block = GRID_RES * GRID_RES * GRID_RES + (start_block + i) % SPARE_MEMORY_SIZE;
		if (atomicCAS(&grid[BLOCK_FREE_SPACE(current_block)], 0, -1) == 0) //claim a blank block of memory
		{
			return current_block;
		}
	}
	return 0;
}
extern "C" __device__ bool assignedToBlockPointedBy(int block, int i, int *grid, int index)
{
	//atomically read which index is the first free one for this grid
	int allocation_block = atomicCAS(&grid[BLOCK(block, i)], 0, 0);
	if (allocation_block == 0)
		return false; //we've reached a null entry
	//otherwise see if this block has space
	int first_free_index = atomicAdd(&grid[BLOCK_FREE_SPACE(allocation_block)], 1) + 1;
	if (first_free_index < BLOCK_SIZE)
	{
		//we have a space
		atomicCAS(&grid[BLOCK(allocation_block, first_free_index)], 0, index);
		return true;
	}
	return false;
}
extern "C" __device__ bool appendListing(int *grid, int memory_listing_block, int new_block, int scramble)
{
	//to add to the grid listings
	//start by finding the current block of listsings
	int next_listing_block = atomicCAS(&grid[NEXT_BLOCK_INDEX(memory_listing_block)], 0, 0);
	while (next_listing_block > 0){
		memory_listing_block = next_listing_block;
		next_listing_block = atomicCAS(&grid[NEXT_BLOCK_INDEX(memory_listing_block)], 0, 0); //no effect, just read
	};
	//get a waiting list number to add a new listings block, just in case
	int new_block_waiting_list = -atomicAdd(&grid[NEXT_BLOCK_INDEX(memory_listing_block)], -1);

	//if we need to add a new listings block
	int block_id = new_block_waiting_list % (BLOCK_SIZE - 1);
	int blocks_wait = new_block_waiting_list / (BLOCK_SIZE - 1);
	blocks_wait = (block_id == 0) ? blocks_wait : blocks_wait + 1;
	//spinlock until the necessary blocks have been allocated
	int allocated_blocks = 0;
	while (allocated_blocks < blocks_wait)
	{
		next_listing_block = atomicMax(&grid[NEXT_BLOCK_INDEX(memory_listing_block)], -1);
		if (next_listing_block > -1)
		{
			memory_listing_block = next_listing_block;
			allocated_blocks++;
		}
	}
	//if we're tasked with allocating the block
	if (block_id == 0)
	{
		//find a new listings block and add it
		int free_listings_block = find_free_block(scramble, grid, scramble);
		atomicCAS(&grid[BLOCK(free_listings_block, 1)], 0, new_block);
		atomicCAS(&grid[BLOCK(free_listings_block, 1)], -1, 0);
		atomicExch(&grid[NEXT_BLOCK_INDEX(memory_listing_block)], free_listings_block);
	}
	else{
		//otherwise set the listing as expected
		atomicCAS(&grid[BLOCK(memory_listing_block, block_id)], 0, new_block);
	}
}
extern "C" __device__ bool new_allocate(int grid_sq, int index, int *grid, int scramble)
{
	int next_listing_block = grid_sq;
	int memory_listing_block = next_listing_block;

	bool allocated = false;
	//start by checking through this particular grid square's listings
	do{
		memory_listing_block = next_listing_block;
		//now check the full listing
		for (int i = 1; i < BLOCK_SIZE; i++)
		{
			if (assignedToBlockPointedBy(memory_listing_block, i, grid, index))
				break;
		}
		next_listing_block = atomicCAS(&grid[NEXT_BLOCK_INDEX(memory_listing_block)], 0, 0); //no effect, just read
	} while (!allocated && next_listing_block != 0);

	//if we need to allocate our own block
	if (!allocated)
	{
		//claim a new chunk of memory
		int free_block = find_free_block(index, grid, scramble);
		//set the first element
		atomicCAS(&grid[BLOCK(free_block, 1)], 0, index);
		//set how much space is left
		atomicCAS(&grid[BLOCK_FREE_SPACE(free_block)], -1, 1);

		//if we can add the listing to this block, do so
		bool added_listing = false;
		for (int i = 1; i < BLOCK_SIZE; i++)
			if (atomicCAS(&grid[BLOCK(memory_listing_block, i)], 0, free_block) == 0)
			{
				added_listing = true;
				break;
			}

		if (!added_listing){
			//appendListing(grid, grid_sq, free_block, scramble*scramble);
		}
	}
}
extern "C" __device__ bool allocate(int grid_sq, int index, int *grid, int scramble)
{
	int memory_block = grid_sq;

	bool allocated_index = allocate_to_block(grid_sq, index, grid, scramble, memory_block);

	//if we run out of space, find a new block
	if (!allocated_index)
	{
		//start by locking the pointer to the old block.
		int prev_point_value = atomicCAS(&grid[memory_block * BLOCK_SIZE + BLOCK_SIZE - 1], 0, -1);
		if (prev_point_value != 0)
			return false; //if we can't get the lock
		
		//random starting position
		int start_block = ((scramble + 1) * index * (SPARE_MEMORY_SIZE - 1) + scramble) % SPARE_MEMORY_SIZE;
		for (int i = 0; i < SPARE_MEMORY_SIZE; i++)
		{
			int current_block = GRID_RES * GRID_RES * GRID_RES + (start_block + i) % SPARE_MEMORY_SIZE;
			if (atomicCAS(&grid[current_block * BLOCK_SIZE], 0, index + 1) == 0) //claim a blank block of memory
			{
				//set the locked pointer to the new block of memory
				atomicCAS(&grid[memory_block * BLOCK_SIZE + BLOCK_SIZE - 1], -1, current_block);
				return true;
			}
		} 
		//unlock block
		atomicCAS(&grid[memory_block * BLOCK_SIZE + BLOCK_SIZE - 1], -1, 0);

		return false;
	}
	return true;

}
extern "C" __global__ void buildAtomicGrid(float *predicted_pos, int *grid, float h,
	float min_x, float min_y, float min_z, int scramble)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT

	//fill up grid  
	if (index < MAX_PARTICLE_COUNT)
	{
		float * pos = &predicted_pos[3 * index];
		int i = (pos[0] - min_x) / h;
		int j = (pos[1] - min_y) / h;
		int k = (pos[2] - min_z) / h;

		bool within_range = (i >= 0 && i < GRID_RES) && (j >= 0 && j <  GRID_RES)
			&& (k >= 0 && k < GRID_RES);
		if (within_range)
		{
			int grid_sq = i * GRID_RES * GRID_RES + j * GRID_RES + k;
			new_allocate(grid_sq, index, grid, scramble);
		}
	}
}
extern "C" __global__ void buildGrid(float *predicted_pos, int *grid, float h,
	float min_x, float min_y, float min_z, int step, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT

	//fill up grid 
	if (index < size)
	{
		float * pos = &predicted_pos[step * index];
		int i = (pos[0] - min_x) / h;
		int j = (pos[1] - min_y) / h;
		int k = (pos[2] - min_z) / h;

		bool within_range = (i >= 0 && i < GRID_RES) && (j >= 0 && j <  GRID_RES)
			&& (k >= 0 && k <  GRID_RES);
		if (within_range)
			grid[index] = i * GRID_RES * GRID_RES + j * GRID_RES + k;
		else
		{
			grid[index] = -1;
			pos[0] = -1000;
		}
	}
}
__device__ void swap_grid(int *array, int i1, int i2)
{
	int temp = array[i1];
	array[i1] = array[i2];
	array[i2] = temp;
}
__device__ void swap_grid_block(int *array, int index, int block_size, int grid_size, bool direction)
{
	//block and block index
	int block = index / block_size;
	int block_i = index % block_size;
	if (block_i * 2 > block_size)
	{
		block += (grid_size / 2) / block_size;
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
extern "C" __global__ void bitonicSortGrid(int *grid, int grid_size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICLE_COUNT/2
	if (!(index < grid_size/2))
		return;

	for (int i = 2; i <= grid_size; i *= 2)
	{
		swap_grid_block(grid, index, i, grid_size, true);
		__syncthreads();

		for (int j = i/2; j > 1; j /= 2)
		{
			swap_grid_block(grid, index, j, grid_size, false);
			__syncthreads();
		}
	}

}
extern "C" __global__ void sortGrid(int *grid, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT
	if (!(index < size))
		return;
	int grid_value = grid[index];

	//mergesort based on the roundedup value of predicted pos, then by index
	int start = index;
	int block_size = 1;
	while (start % (block_size * 2) == 0 && block_size < size)
	{
		int i = 0;
		int j = 0;
		int w = 0;
		int* current_block = &grid[2 * start];

		//merge sort this section
		while (i < block_size && j < block_size && j + start < size)
		{
			if (current_block[i] < current_block[(block_size + j)])
				swap_grid(current_block, w++, i++);
			else
				swap_grid(current_block, w++, block_size + j++);
		}

		while (i < block_size)
			swap_grid(current_block, w++, i++);

		while (j < block_size && j + block_size + start < size)
			swap_grid(current_block, w++, block_size + j++);


		block_size *= 2;
		__syncthreads();
	}

}
extern "C" __global__ void buildIndex(int *grid, int *grid_index, int grid_size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x; //thread id, from 0 to MAX_PARTICL E_COUNT
	if (!(index < GRID_RES*GRID_RES*GRID_RES))
		return;

	//search for grid value, where each cell is a rounded up predicted pos value
	int start_index = grid_size / 2;
	int end_index = grid_size / 2;
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
extern "C" __global__ void findNeighboursAtomicGrid(float *predicted_pos, int *grid, int scramble,
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

	bool within_range = (i_i >= 0 && i_i < GRID_RES) && (i_j >= 0 && i_j < GRID_RES)
		&& (i_k >= 0 && i_k < GRID_RES);
	if (!within_range)
		return;

	//find neighbours`
	int found_neighbours = 0;
	float h_2 = h*h;
	//loop through neighbouring squares
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
			{
				int r_i = ((i*scramble + scramble) % 3 - 1 + i_i + GRID_RES) % GRID_RES;
				int r_j = ((j*scramble + scramble) % 3 - 1 + i_j + GRID_RES) % GRID_RES;
				int r_k = ((k*scramble + scramble) % 3 - 1 + i_k + GRID_RES) % GRID_RES;

				int grid_offsetted_val = r_i * GRID_RES * GRID_RES + r_j * GRID_RES + r_k;
				int counter = grid_offsetted_val * BLOCK_SIZE;
				int offset = grid[counter] - 1; //start with the block allocated for this grid cell
				offset = (r_i*index*scramble + scramble*r_k + scramble*r_i) % MAX_PARTICLE_COUNT;
				if (offset < 0)
					continue; //skip empty cells
				do{ 
					//check all the neighbouring particles in this 
					float distance = 0;
					for (int i = 0; i < 3; i++)
						distance += (pos[i] - predicted_pos[3 * offset + i]) * (pos[i] - predicted_pos[3 * offset + i]);
					//if its close enough
					if (distance < h_2)
					{
						neighbour_indexes[index*MAX_NEIGHBOURS + found_neighbours] = offset;
						found_neighbours++;
					}
					 
					//move onto next pos
					counter++;
					//if we reach the endof a block, usre the referenceto the next omne
					if (counter % BLOCK_SIZE == BLOCK_SIZE - 1)
						counter = (grid[counter] * BLOCK_SIZE);
					offset = grid[counter] - 1;
					 
				} while (counter > 0 && offset > 0 && found_neighbours < MAX_NEIGHBOURS);
			}
	//fill up the rest of the neighbours
	while (found_neighbours < MAX_NEIGHBOURS)
	{
		neighbour_indexes[index*MAX_NEIGHBOURS + found_neighbours] = -1;
		found_neighbours++;
	}
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

extern "C" __device__  void cross(float* p, float* q, float* result)
{
	//basic cross product
	result[0] = p[2] * q[1] - p[1] * q[2];
	result[1] = p[0] * q[2] - p[2] * q[0];
	result[2] = p[1] * q[0] - p[0] * q[1];
}
extern "C" __device__  void add(float* p, float* q, float* output)
{
	for (int i = 0; i < 3; i++)
		output[i] = p[i] + q[i];
}
extern "C" __device__  void multiply(float* p, float q, float* output)
{
	for (int i = 0; i < 3; i++)
		output[i] = p[i] * q;
}
extern "C" __device__  float dot(float* p, float* q)
{
	return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
}
extern "C" __device__  void lerp(float* p, float* q, float r, float* output)
{
	for (int i = 0; i < 3; i++)
		output[i] = p[i] * r + q[i] * (1 - r);
}
extern "C" __device__  void set(float* p, float* q)
{
	for (int i = 0; i < 3; i++)
		q[i] = p[i];
}
extern "C" __device__  bool in_triangle(float* p, float* p0, float* p1, float* p2)
{
	// Compute vectors
	float v0[3];
	float v1[3];
	float v2[3];
	for (int i = 0; i < 3; i++)
	{
		v0[i] = p2[i] - p0[i];
		v1[i] = p1[i] - p0[i];
	}
	//normal vector
	float n[3];
	cross(v0, v1, n);
	float d = (p[0] - p0[0])*n[0] + (p[1] - p0[1])*n[1] + (p[2] - p0[2])*n[2];

	for (int i = 0; i < 3; i++)
	{
		v2[i] = (p[i] - d*n[i]) - p0[i];
	}

	// Compute dot products
	float dot00 = dot(v0, v0);
	float dot01 = dot(v0, v1);
	float dot02 = dot(v0, v2);
	float dot11 = dot(v1, v1);
	float dot12 = dot(v1, v2);

	// Compute barycentric coordinates
	float denom = (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12);
	float v = (dot00 * dot12 - dot01 * dot02) ;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < denom);
}
extern "C" __device__  bool triCollide(float* normal, float* tri_a, float* tri_b, float* tri_c,
	float* velocity, float* pos, float* next_pos, float delta)
{
	float plane_height = dot(tri_a, normal);
	float pos_height_before = dot(pos, normal);
	float pos_height_after = dot(next_pos, normal);

	//if particle position before and after the timestep are on opposite sides of the plane 
	if ((pos_height_after - plane_height)* (pos_height_before - plane_height) < 0)
	{
		//calculate intersection point
		float intersection_point[3];
		float interpolation = (plane_height - pos_height_before) / (pos_height_after - pos_height_before);
		lerp(pos, next_pos, interpolation, intersection_point);

		//check if we're inside the triangle
		if (in_triangle(intersection_point, tri_a, tri_b, tri_c))
		{
			float perpendicular_component = dot(velocity, normal);

			//reflect of the plane
			float impulse[3];
			multiply(normal, -2 * perpendicular_component, impulse);
			add(velocity, impulse, velocity);

			//move the particle back over the boundary
			multiply(impulse, delta, impulse);
			add(intersection_point, impulse, intersection_point);
			set(next_pos, intersection_point);

			return true;
		}
		return false;
	}
	return false;
}
extern "C" __device__  void collisions(float* speed, float* pos, float* next_pos, float* life, float delta,
	int* collide_grid, float* collide_data, float min_x, float min_y, float min_z, float h,
	float* matrix)
{	
	float transformed_pos[] = { 0, 0, 0 };
	float transformed__next_pos[] = { 0, 0, 0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			transformed_pos[i] += matrix[4 * i + j] * pos[j];
			transformed__next_pos[i] += matrix[4 * i + j] * next_pos[j];
		}


		transformed_pos[i] += matrix[4 * i + 4];
		transformed__next_pos[i] += matrix[4 * i + 4];
	}


	//find grid cell
	int i_i = (pos[0] - min_x) / h;
	int i_j = (pos[1] - min_y) / h;
	int i_k = (pos[2] - min_z) / h;

	bool within_range = (i_i >= 0 && i_i < GRID_RES) && (i_j >= 0 && i_j < GRID_RES)
		&& (i_k >= 0 && i_k < GRID_RES);
	if (!within_range)
		return;

	int grid_index = i_i * GRID_RES * GRID_RES + i_j * GRID_RES + i_k;

	int counter = grid_index * BLOCK_SIZE;
	int offset = collide_grid[counter] - 1;; //start with the block allocated for this grid cell
	//offset = 0;
	while (offset < 12)
	{
		//check all the neighbouring particles in this 
		float* collide_tri_pos = &collide_data[offset * 3 * 4];
		float* collide_tri_norm = collide_tri_pos + 3;
		float* collide_tri_dir_a = collide_tri_pos + 6;
		float* collide_tri_dir_b = collide_tri_pos + 9;

		triCollide(collide_tri_norm, collide_tri_pos, collide_tri_dir_a, collide_tri_dir_b,
			speed, transformed_pos, transformed__next_pos, delta);


		//move onto next pos
		counter++;
		//if we reach the endof a block, usre the referenceto the next omne
		if (counter % BLOCK_SIZE == BLOCK_SIZE - 1)
			counter = (collide_grid[counter] * BLOCK_SIZE);
		offset = collide_grid[counter] - 1;
	}
	for (int i = 0; i < 3; i++)
	{
		transformed_pos[i] -= matrix[4 * i + 4];
		transformed__next_pos[i] -= matrix[4 * i + 4];

		pos[i] = 0;
		next_pos[i] = 0;

		for (int j = 0; j < 3; j++)
		{
			pos[i] += matrix[4 * j + i] * transformed_pos[j];
			next_pos[i] += matrix[4 * j + i] * transformed__next_pos[j];
		}

	}

}
extern "C" __device__  void applyForce(float* speed, float* pos, float* next_pos, float* life, float delta, 
	 int* collide_grid, float* collide_data, float min_x, float min_y, float min_z, float h,
	 float* matrix, float viscocity, float buoyancy)
{
	//uniform force + boyancy
	speed[0] += 0;
	speed[1] += -9.81 * delta - (life[0] - 2.5f)*buoyancy;
	speed[2] += 0;
	
	//viscous drag
	for (int i = 0; i < 3; i++)
		speed[i] *= viscocity;

	//texture lookup force
	uchar4 velocity = tex2D(external_force, (pos[0] - min_x) / (h * GRID_RES), (pos[1] - min_y) / (h * GRID_RES));
	speed[0] += velocity.x * delta;
	speed[1] += velocity.y * delta;
	speed[2] += velocity.z * delta;

	collisions(speed, pos, next_pos, life, delta, collide_grid, collide_data, min_x, min_y, min_z, h, matrix);
};
extern "C" __global__ void updateParticles(float delta, float *pos, float* predicted_pos, float* speed,
	float* life, GLfloat* GL_positions, GLubyte* GL_colors, int scramble, int* collide_grid, float* collide_data,
	float* matrix, float* settings)
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

			GL_colors[4 * i] = speed[3 * i] + 128 + (char)((5.0 - life[i]) * 128);
			GL_colors[4 * i + 1] = speed[3 * i + 1] + 128;
			GL_colors[4 * i + 2] = speed[3 * i + 2] + 128;
			GL_colors[4 * i + 3] = 255;

			// update speed, 
			for (int j = 0; j < 3; j++)
				speed[3 * i + j] = (predicted_pos[3 * i + j] - pos[3 * i + j])*(1 / delta);
			//apply physics force
			applyForce(&speed[3 * i], &pos[3 * i], &predicted_pos[3 * i], &life[i], delta, 
				 collide_grid, collide_data,GRID_X, GRID_Y, GRID_Z, 5, matrix, settings[0], settings[1]);
			//for next frame
			for (int j = 0; j < 3; j++)
			{
				pos[3 * i + j] = predicted_pos[3 * i + j];
				predicted_pos[3 * i + j] = pos[3 * i + j] + speed[3 * i + j] * delta;
			}
			//kill particles out of the viewing frustrum
			int i_i = (pos[0] - GRID_X) / 5;
			int i_j = (pos[1] - GRID_Y) / 5;
			int i_k = (pos[2] - GRID_Z) / 5;

			bool within_range = (i_i >= 0 && i_i < GRID_RES) && (i_j >= 0 && i_j < GRID_RES)
				&& (i_k >= 0 && i_k < GRID_RES);
			if (!within_range)
				life[i] = -1;
		}
		else
		{
			// move it out of the viewing frustrum
			GL_positions[4 * i] = -1000;
			if ((i * scramble) % 103 < 20) //one half of the remaining particles are brought to life
			{

				double theta = ((i * scramble) % 628)*0.01;
				double phi = ((i * scramble * scramble) % 314)*0.005;

				pos[3 * i] = sin(phi)*sin(theta) * 10;
				pos[3 * i + 1] = cos(phi) * 10 + 50;
				pos[3 * i + 2] = sin(phi)*cos(theta) * 10;

				speed[3 * i] =  ((scramble * i) % 100 - 100) * 0.1;
				speed[3 * i + 1] =  -(scramble * i * i * i) % 100 - 150;
				speed[3 * i + 2] = ((scramble * i * i) % 100 - 50) * 0.1;

				for (int j = 0; j < 3; j++)
					predicted_pos[3 * i + j] = pos[3 * i + j] + speed[3 * i + j] * delta;

				life[i] = 0.5f; //lasts 5 second
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
	if (!(particle_index < MAX_PARTICLE_COUNT))
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
		for (int j = 0; j < MAX_NEIGHBOURS; j++)
		{
			predicted_pos[3 * particle_index] += predicted_pos_contributions[storage_index + j].x;
			predicted_pos[3 * particle_index + 1] += predicted_pos_contributions[storage_index + j].y;
			predicted_pos[3 * particle_index + 2] += predicted_pos_contributions[storage_index + j].z;
		}
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
	if (!(particle_index < MAX_PARTICLE_COUNT))
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
		lambda[particle_index] = -numerator / denominator;
	}
} 
void ParticleContainer::setForceTexture(unsigned char* tex, int width, int height)
{
	//Allocate 2D memory on GPU. Also known as Pitch Linear Memory
	uchar4* cuda_tex;
	size_t gpu_image_pitch = 0;
	width = 64;
	height = 64;
	gpuErrchk(cudaMallocPitch<uchar4>(&cuda_tex, &gpu_image_pitch, width, height));
	//Copy data from host to device.
	gpuErrchk(cudaMemcpy2D(cuda_tex, gpu_image_pitch, tex, width, width, height, cudaMemcpyHostToDevice));
	//Bind the image to the texture =
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(8,8,8,8, cudaChannelFormatKindUnsigned);
	gpuErrchk(cudaBindTexture2D(NULL, &external_force, cuda_tex, &desc, width, height, gpu_image_pitch));
}
void ParticleContainer::setMatrix(float* mat)
{
	size_t mem_size = sizeof(float) * 16;
	gpuErrchk(cudaMemcpy(matrix_CUDA, mat, mem_size, cudaMemcpyHostToDevice));
}
void ParticleContainer::setConstants(float v, float b)
{
	float constants[16];
	constants[0] = v;
	constants[1] = b;
	size_t mem_size = sizeof(float) * 16;
	gpuErrchk(cudaMemcpy(constants_CUDA, constants, mem_size, cudaMemcpyHostToDevice));
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

#ifdef ATOMIC_METHOD
	mem_size = sizeof(int) * (SPARE_MEMORY_SIZE + GRID_RES * GRID_RES * GRID_RES)* BLOCK_SIZE;
#else
	mem_size = sizeof(int) * MAX_PARTICLE_COUNT;
#endif
	gpuErrchk(cudaMalloc((void **)&grid_CUDA, mem_size));
	mem_size = sizeof(int) * GRID_RES * GRID_RES * GRID_RES * 2;
	gpuErrchk(cudaMalloc((void **)&grid_index_CUDA, mem_size));

	//for copying to renderbuffers
	mem_size = sizeof(GLfloat) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMalloc((void **)&positions_CUDA, mem_size));
	gpuErrchk(cudaMemset((void *)positions_CUDA,0, mem_size));

	mem_size = sizeof(GLubyte) * MAX_PARTICLE_COUNT * 4;
	gpuErrchk(cudaMalloc((void **)&colors_CUDA, mem_size));

	//for collisions
	mem_size = sizeof(float) * MAX_FACE_COUNT * 4 * 3;
	gpuErrchk(cudaMalloc((void **)&collision_data_CUDA, mem_size));
	mem_size = sizeof(int) * (SPARE_MEMORY_SIZE + GRID_RES * GRID_RES * GRID_RES)* BLOCK_SIZE;
	gpuErrchk(cudaMalloc((void **)&collision_grid_CUDA, mem_size));
	mem_size = sizeof(float) * 16;
	gpuErrchk(cudaMalloc((void **)&matrix_CUDA, mem_size));

	mem_size = sizeof(float) * 16;
	gpuErrchk(cudaMalloc((void **)&constants_CUDA, mem_size));

	//init data
	updateParticles << < 1, 512 >> >(0.2, particle_positions_CUDA,
		particle_predicted_pos_CUDA, particle_speeds_CUDA, particle_life_CUDA,
		positions_CUDA, colors_CUDA, 1, collision_grid_CUDA, collision_data_CUDA, matrix_CUDA, constants_CUDA);
	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
}
void ParticleContainer::cleanup_CUDA()
{
	gpuErrchk(cudaFree((void **)&particle_positions_CUDA));
	gpuErrchk(cudaFree((void **)&particle_lambdas_CUDA));
	gpuErrchk(cudaFree((void **)&particle_speeds_CUDA)) ;
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
		positions_CUDA, colors_CUDA, rand(), collision_grid_CUDA, collision_data_CUDA, matrix_CUDA, constants_CUDA);

	//Error handling
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
}
void ParticleContainer::findNeighboursAtomic_CUDA(float delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	//find neigbours for next iteration
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;

	size_t mem_size = sizeof(int)*(SPARE_MEMORY_SIZE + GRID_RES * GRID_RES * GRID_RES) * BLOCK_SIZE;
	gpuErrchk(cudaMemset((void*)grid_CUDA, 0, mem_size));

	buildAtomicGrid << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA, grid_CUDA,
		5, GRID_X, GRID_Y, GRID_Z, rand());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Build atomic grid  %d ms \n");

	//mem_size = GRID_RES*GRID_RES*GRID_RES*sizeof(int);
	//gpuErrchk(cudaMemcpy(grid_array, grid_CUDA, mem_size, cudaMemcpyDeviceToHost));

	blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	findNeighboursAtomicGrid << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA, grid_CUDA,
		rand(), neighbours_CUDA, GRID_X, GRID_Y, GRID_Z, 5);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());

	RECORD_SPEED("		Use atomic grid to find particles  %d ms \n");
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
			particle_life_CUDA, particle_speeds_CUDA, particle_lambdas_CUDA, neighbours_CUDA, 0.5, 15, 0.1);

		//Error handling
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaGetLastError());

		// Launch the CUDA Kernel for positions
		solverIterationPositions << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA,
			particle_life_CUDA, particle_lambdas_CUDA, neighbours_CUDA, 15, 0.0121611953, 0.1, 0.5, 4);
		//Error handling
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaGetLastError());
	}
}
void buildCollisionGridCPU(float *collision_data, int *grid, float h, float min_x, float min_y, float min_z, int scramble, int size)
 {
	int next_block = GRID_RES*GRID_RES*GRID_RES;

	for (int index = 0; index < size; index++)
	 {
		 float * pos = &collision_data[3 * 4 * index];
		 float * dir1 = &collision_data[3 * 4 * index + 6];
		 float * dir2 = &collision_data[3 * 4 * index + 9];

		 float scale_factor_1 = 1 / ((max(abs(dir1[0]), max(abs(dir1[1]), abs(dir1[2]))))*h);
		 float length_1 = 1.0f/sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + dir1[2] * dir1[2]); //already invcerted

		 float scale_factor_2 = 1 / ((max(abs(dir2[0]), max(abs(dir2[1]), abs(dir2[2])))) * h);
		 float length_2 = 1.0f/sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + dir2[2] * dir2[2]);

		 for (float a = 0; a < length_1*length_1; a += scale_factor_1*length_1)
			 for (float b = 0; b < length_2*length_2; b += scale_factor_2*length_2)
			 {
				 float new_pos[3];
				 new_pos[0] = pos[0] + a*dir1[0] + b*dir2[0];
				 new_pos[1] = pos[1] + a*dir1[1] + b*dir2[1];
				 new_pos[2] = pos[2] + a*dir1[2] + b*dir2[2];

				 int i = (new_pos[0] - min_x) / h;
				 int j = (new_pos[1] - min_y) / h;
				 int k = (new_pos[2] - min_z) / h;

				 bool within_range = (i >= 0 && i < GRID_RES) && (j >= 0 && j < GRID_RES)
					 && (k >= 0 && k <  GRID_RES);
				 if (within_range)
				 {
					 int grid_sq = i * GRID_RES * GRID_RES + j * GRID_RES + k;

					 //follow the chain to the correct block
					 int allocated_block = grid_sq;
					 while (grid[allocated_block*BLOCK_SIZE + BLOCK_SIZE - 1] > 0 && allocated_block < 
						 SPARE_MEMORY_SIZE + GRID_RES * GRID_RES * GRID_RES){
						 allocated_block = grid[allocated_block*BLOCK_SIZE + BLOCK_SIZE - 1];
					 }

					 //search for a free square
					 bool found = false;
					 for (int i = 1; i < BLOCK_SIZE - 1; i++)
						 if (grid[allocated_block*BLOCK_SIZE + i] == 0)
						 {
							grid[allocated_block*BLOCK_SIZE + i] = index + 1;
							 found = true;
							 break;
						 }

					 //allocate a new block
					 if (!found)
					 {
						 grid[allocated_block*BLOCK_SIZE + BLOCK_SIZE - 1] = next_block;
						 grid[next_block*BLOCK_SIZE] = index + 1;
						 next_block++;
					 }
			 }
		}
	 }
 }
void ParticleContainer::loadModel_CUDA()
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	int n;
	float* collision_data = mesh->data(n);
	gpuErrchk(cudaMemcpy(collision_data_CUDA, collision_data, n * 12 * sizeof(float), cudaMemcpyHostToDevice));

	int mem_size = (SPARE_MEMORY_SIZE + GRID_RES*GRID_RES*GRID_RES) * BLOCK_SIZE * sizeof(int);
	memset(collision_grid_array, 0, mem_size);
	buildCollisionGridCPU(collision_data, collision_grid_array, 5, GRID_X, GRID_Y, GRID_Z, rand(), n);

	gpuErrchk(cudaMemcpy(collision_grid_CUDA, collision_grid_array, mem_size, cudaMemcpyHostToDevice));
	RECORD_SPEED("		Build colllision grid  %d ms \n");
}

void ParticleContainer::findNeighbours_CUDA(float delta)
{
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	//find neigbours for next iteration
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	buildGrid << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA, grid_CUDA,
		5, GRID_X, GRID_Y, GRID_Z, 3, MAX_PARTICLE_COUNT);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Build grid  %d ms \n");

	//sort the grid
	blocksPerGrid = GRID_RES * GRID_RES * GRID_RES / THREADS_PER_BLOCK + 1;
	bitonicSortGrid << <blocksPerGrid, threadsPerBlock >> >(grid_CUDA, GRID_RES * GRID_RES * GRID_RES);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Sort grid  %d ms \n");

	//build index
	buildIndex << <blocksPerGrid, threadsPerBlock >> >(grid_CUDA, grid_index_CUDA, MAX_PARTICLE_COUNT);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Build index  %d ms \n");

	int mem_size = GRID_RES*GRID_RES*sizeof(int);
	gpuErrchk(cudaMemcpy(grid_array, grid_CUDA, mem_size, cudaMemcpyDeviceToHost));

	//find neighbours
	blocksPerGrid = MAX_PARTICLE_COUNT / THREADS_PER_BLOCK + 1;
	findNeighboursGrid << <blocksPerGrid, threadsPerBlock >> >(particle_predicted_pos_CUDA, grid_CUDA, 
		grid_index_CUDA, rand(), neighbours_CUDA, GRID_X, GRID_Y, GRID_Z, 5);
	gpuErrchk(cudaDeviceSynchronize()); 
	gpuErrchk(cudaGetLastError());
	RECORD_SPEED("		Use grid to find particles  %d ms \n");


	mem_size = max_particle_count * MAX_NEIGHBOURS*sizeof(int);
	gpuErrchk(cudaMemcpy(neighbours_array, neighbours_CUDA, mem_size, cudaMemcpyDeviceToHost));
	mem_size = GRID_RES*GRID_RES*sizeof(int);
	gpuErrchk(cudaMemcpy(grid_array, grid_CUDA, mem_size, cudaMemcpyDeviceToHost));

} 