#ifndef PERFORMANCE_SETTINGS
#define PERFORMANCE_SETTINGS
#include <cstdio>
#define RECORD_SPEED(x) time = glutGet(GLUT_ELAPSED_TIME); printf(x, time-start_time); start_time = time;
//#define RECORD_SPEED(x) time = glutGet(GLUT_ELAPSED_TIME); fprintf(csv, "%f,",time-start_time); start_time = time;
#define SPACE() fprintf(csv, "\n");

#define ATOMIC_METHOD
#define BLOCK_SIZE 8
//#define USE_KDTREE
//#define USE_CUDA
#define SPARE_MEMORY_SIZE MAX_PARTICLE_COUNT*2
#define MAX_NEIGHBOURS 8
#define GRID_RES 32
#define GRID_X -50
#define GRID_Y -75
#define GRID_Z -50
#define THREADS_PER_BLOCK 512
#define MAX_THREADS 512
#define EPSILON 0.00001
#define MAX_FACE_COUNT 9000
#define MAX_PARTICLE_COUNT 12000
#define max_particle_count MAX_PARTICLE_COUNT

#ifdef __cuda_cuda_h__
#ifndef GPU_ERROR
#define GPU_ERROR
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#endif
#endif
#endif


