#ifndef PERFORMANCE_SETTINGS
#define PERFORMANCE_SETTINGS

#define RECORD_SPEED(x) time = glutGet(GLUT_ELAPSED_TIME); printf(x, time-start_time); start_time = time;

#define USE_CUDA
#ifdef USE_CUDA
#define USE_KDTREE
#define MAX_NEIGHBOURS 32
#define MAX_NEIGHBOURS_LOG 3
#define THREADS_PER_BLOCK 256
#define MAX_THREADS 768
#define EPSILON 0.00001
#endif

#define MAX_PARTICLE_COUNT 4096*2
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


