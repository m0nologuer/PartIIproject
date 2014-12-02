#pragma once
// CPU representation of a particle

#include <math.h>
#include <cstdlib>
#include <vector>

using namespace std;

#include <GL/glut.h>
#include <GL/glext.h>

struct Particle{
	struct vec3{
		double x;
		double y;
		double z;
		vec3(){};
		vec3(double _x, double _y, double _z):x(_x),y(_y),z(_z) {};

		vec3 operator*(double f)
		{
			vec3 result;
			result.x = this->x *f;
			result.y = this->y *f;
			result.z = this->z *f;
			return result;
		}
		vec3 operator+(vec3& y)
		{
			vec3 result;
			result.x = this->x + y.x;
			result.y = this->y + y.y;
			result.z = this->z + y.z;
			return result;
		}
		vec3 operator-(vec3& y)
		{
			vec3 result;
			result.x = this->x - y.x;
			result.y = this->y - y.y;
			result.z = this->z - y.z;
			return result;
		}

		static double dot(vec3 x, vec3 y)
		{
			return x.x*y.x + x.y*y.y + x.z*y.z;
		}
	};

	vec3 pos, speed, predicted_pos, d_p_pos;
	unsigned char r, g, b, a; // Color
	double lambda;
	double size, angle, weight;
	double life; // Remaining life of the particle. if < 0 : dead and unused.
	double cameradistance; //for sorting
};
#define h_squared (h*h)
#define h_6 (h_squared*h_squared*h_squared)
#define h_9 (h_squared*h_squared*h_squared*h_squared*h)


class ParticleContainer
{

public:
	//constants
	static const int max_particle_count = 2000;
	static const int n = 4;

	float Wq;
	float h;
	float corr_k;
	float q;
	float e0;
	float p0;

	//pointers to GL buffers
	GLuint particles_color_buffer;
	GLuint particles_position_buffer;
	GLuint billboard_vertex_buffer;

	ParticleContainer();
	~ParticleContainer();

	void Init(char* texture, char* vertexShader, char* pixelShader);
	void UpdateParticles(double delta);
	void Draw();

	int getParticleCount();
	double getAverageSpeed();

private:
	static const int particles_per_second = 40;
	static const int iteration_count = 45;

	Particle container[max_particle_count];

	//for rendering
	Particle::vec3 camera_pos;
	int last_particle_count; //search for next available particle
	int render_counter; //umber of particles to be rendered

	//for physics
	vector<int> neighbours[max_particle_count];

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
	Particle::vec3 getParticleForce(Particle::vec3 postion);
	vector<int> findNeighbouringParticles(Particle postion);
	Particle::vec3 collisionUpdate(int index);

	//constraint functions
	double constraint_function(int index);
	double gradient_constraint_function(int i, int k);
	double lambda(int index);

	//interpolation kernels
	double W_spiky(Particle::vec3 r);
	double W_poly6(Particle::vec3 r);
	Particle::vec3 dW_spiky(Particle::vec3 r);
	Particle::vec3 dW_poly6(Particle::vec3 r);

	//create GPU buffers;
	void initializeBuffers();
};

