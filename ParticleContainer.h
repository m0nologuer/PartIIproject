#pragma once
// CPU representation of a particle

#include <math.h>
#include <cstdlib>
#include <vector>

using namespace std;

#include <GL/glut.h>log_speed
#include <GL/glext.h>
#include "KDTree.h"
#include "PerformanceSettings.h"
#include "COLLADALoader.h"

#define h_squared (h*h)
#define h_6 (h_squared*h_squared*h_squared)
#define h_9 (h_squared*h_squared*h_squared*h_squared*h)

#define USE_KDTREE 

class ParticleContainer
{
public:
	//constants
	static const int n = 4;

	//pointers to GL buffers
	GLuint particles_color_buffer;
	GLuint particles_position_buffer;
	GLuint billboard_vertex_buffer;

	ParticleContainer();
	~ParticleContainer();

	void Init(char* texture, char* vertexShader, char* pixelShader);
	void SetObstacle(ColladaLoader* mesh);
	void UpdateParticles(double delta);
	void Draw();

	//information getters
	int getParticleCount();
	double getAverageSpeed();
	std::string livePositionsList();
	std::vector<Particle> getAll();

private:
#ifdef USE_CUDA
	//CUDA accelerated physics
	void intialize_CUDA();
	void solverIterations_CUDA();
	void cleanup_CUDA();

	//CUDA variables
	Particle* container_CUDA;
	int* neighbours_CUDA;
	int* neighbour_array;
#endif

	static const int particles_per_iteration = 200;
	static const int iteration_count = 25;
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

	//for rendering
	Particle::vec3 camera_pos;
	int last_particle_count; //search for next available particle
	int render_counter; //umber of particles to be rendered

	//for physics
	vector<Particle*> neighbours[max_particle_count];

	//GPU data
	GLfloat g_particle_position_data[4 * max_particle_count];
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

