#ifndef PERFORMANCE_SETTINGS
#define PERFORMANCE_SETTINGS
#include <cstdio>
#define CSV_NEWL() fprintf(csv, "\n");
#define RECORD_SPEED(x) time = glutGet(GLUT_ELAPSED_TIME); printf(x, time-start_time); start_time = time;

//#define ATOMIC_METHOD
#define MAX_NEIGHBOURS 16
#define GRID_RES 16 
#define THREADS_PER_BLOCK 512
#define MAX_THREADS 512
#define EPSILON 0.00001

#define MAX_PARTICLE_COUNT 6000
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


