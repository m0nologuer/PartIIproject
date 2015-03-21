#pragma once
// CPU representation of a particle

#include <math.h>
#include <cstdlib>
#include <vector>

using namespace std;

#include <GL/glut.h>
#include <GL/glext.h>
#include "KDTree.h"
#include "PerformanceSettings.h"
#include "COLLADALoader.h"

#define h_squared (h*h)
#define h_6 (h_squared*h_squared*h_squared)
#define h_9 (h_squared*h_squared*h_squared*h_squared*h)

struct Config{
	int particle_count;
	int neighbour_count;
	bool use_CUDA;
	bool use_KDTree;
};

class ParticleContainer
{
public:
	//constants
	static const int n = 4;
	FILE* csv;

	//pointers to GL buffers
	GLuint particles_color_buffer;
	GLuint particles_position_buffer;
	GLuint billboard_vertex_buffer;

	ParticleContainer();
	~ParticleContainer();

	void Init(char* texture, char* vertexShader, char* pixelShader, char* force_texture);
	void Set(ColladaLoader* mesh, Config* settings);
	void UpdateParticles(double delta);
	void Draw();
	void DrawPoints();

	void setMatrix(float* inverse_mat);
	void setConstants(float v, float b);
	void setForceTexture(unsigned char* tex, int width, int height);

	//information getters
	int getParticleCount();
	double getAverageSpeed();
	std::string livePositionsList();
	std::vector<Particle> getAll();

private:
	//CUDA accelerated physics
	void CUDAloop(double delta);
	void loadModel_CUDA();
	void updateParticles_CUDA(double delta);
	void findNeighbours_CUDA(float delta); 
	void findNeighboursAtomic_CUDA(float delta);
	void intialize_CUDA();
	void solverIterations_CUDA(float delta);
	void cleanup_CUDA();

	//CUDA variables
	GLfloat* positions_CUDA;
	GLubyte* colors_CUDA;

	//for particle movement & solvers
	float* particle_positions_CUDA;
	float* particle_speeds_CUDA;
	float* particle_lambdas_CUDA;
	float* particle_life_CUDA;
	float* particle_predicted_pos_CUDA;

	//determine neihgbours
	int* neighbours_CUDA;
	int* grid_CUDA;
	int* grid_index_CUDA;

	//settings
	float* constants_CUDA;

	//for collisions
	float* collision_data_CUDA;
	int* collision_grid_CUDA;
	float* matrix_CUDA;

	//for testing
	int collision_grid_array[(SPARE_MEMORY_SIZE + GRID_RES*GRID_RES*GRID_RES) * BLOCK_SIZE];
	int grid_array[(SPARE_MEMORY_SIZE + GRID_RES*GRID_RES*GRID_RES) * BLOCK_SIZE];
	int neighbours_array[max_particle_count * MAX_NEIGHBOURS];
	float lambdas_array[max_particle_count];

	static const int particles_per_iteration = 1000;
	static const int iteration_count = 10;
	static const int life = 5.0f;

	float Wq;
	float h;
	float corr_k;
	float q;
	float e0;
	float p0;
 
	char colors[5][3];

	Particle container[max_particle_count];
	KDTree* tree;
	ColladaLoader* mesh;
	Config settings;

	//for rendering
	Particle::vec3 camera_pos;
	int last_particle_count; //search for next available particle
	int render_counter; //umber of particles to be rendered

	//for physics
	vector<Particle*> neighbours[max_particle_count];

	//GPU data
	GLfloat g_particle_position_data[4 * max_particle_count];
	GLfloat g_particle_velocity_data[4 * max_particle_count];
	GLubyte g_particle_color_data[4 * max_particle_count];

	//shader and texture variables
	GLuint programID;
	GLuint textureID;
	GLuint viewVec1ID, viewVec2ID, vpMatID;
	GLuint billboardTextureID;

	//position and transformation
	float viewMat[4][4];
	float projMat[4][4];

	//updating particles
	double average_speed;
	void addNewParticles(double delta);
	void applyPhysics(double delta);
	int getUnusedParticle();
	void updateGLBuffers();

	//physics helper functions
	void solverIteration();
	Particle::vec3 getParticleForce(Particle postion);
	vector<Particle*> findNeighbouringParticles(Particle postion);
	Particle::vec3 collisionUpdate(int index);

	//constraint functions
	double constraint_function(int index);
	double gradient_constraint_function(int index, Particle* pk);
	double lambda(int index);

	//interpolation kernels
	double W_spiky(Particle::vec3 r);
	double W_poly6(Particle::vec3 r);
	Particle::vec3 dW_spiky(Particle::vec3 r);
	Particle::vec3 dW_poly6(Particle::vec3 r);

	//create GPU buffers;
	void initializeBuffers();
};

