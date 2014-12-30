#include "ParticleContainer.h"

#include <stdio.h>

ParticleContainer::ParticleContainer()
{
	last_particle_count = 0;
	for (int i = 0; i < max_particle_count; i++)
	{
		container[i].life = -1; //initalize all particles dead
	}
	average_speed = 0;

	//for the correction 
	h = 5; //distance paramter
	e0 = 0.1; //relaxion parameter
	corr_k = 0.1;
	q = 0.1*h;
	p0 = 0.5; //rest desngity

	Wq = W_poly6(Particle::vec3(q,0,0));

	particles_color_buffer = NULL;
	particles_position_buffer = NULL;
	tree = NULL;

	addNewParticles(1); //initialize with a lot of particles

	//hard code colors
	colors[0][0] = 83; colors[0][1] = 119; colors[0][2] = 122;
	colors[1][0] = 84; colors[1][1] = 36; colors[1][2] = 55;
	colors[2][0] = 192; colors[2][1] = 41; colors[2][2] = 66;
	colors[3][0] = 217; colors[2][1] = 91; colors[2][2] = 67;
	colors[3][0] = 236; colors[2][1] = 208; colors[2][2] = 120;
}
ParticleContainer::~ParticleContainer()
{
}
int ParticleContainer::getParticleCount(){
	return render_counter;
}
double ParticleContainer::getAverageSpeed(){
	return average_speed;
}
void ParticleContainer::SetObstacle(ColladaLoader* m)
{
	mesh = m;
}
std::string ParticleContainer::livePositionsList()
{
	std::string output = "x,y,z\n";
	//create a list ofpositions of live particles
	for (int i = 0; i < max_particle_count; i++)
		if (container[i].life > 0)
		{
			char buffer[512];
			Particle::vec3 p = container[i].pos;
			sprintf((char*)buffer, "%f, %f, %f \n", p.x, p.y, p.z);
			output += buffer;
		}
	return output;
}
std::vector<Particle> ParticleContainer::getAll()
{
	std::vector<Particle> output;
	for (int i = 0; i < max_particle_count; i++)
		output.push_back(container[i]);
	return output;
}
int ParticleContainer::getUnusedParticle(){

	for (int i = last_particle_count; i< max_particle_count; i++){
		if (container[i].life < 0){
			last_particle_count = i;
			return i;
		}
	}

	for (int i = 0; i< last_particle_count; i++){
		if (container[i].life < 0){
			last_particle_count = i;
			return i;
		}
	}

	return 0; // All particles are taken, override the first one
}
void ParticleContainer::addNewParticles(double delta)
{
	//specified number of particles per time period;

	for (int i = 0; i < particles_per_iteration; i++)
	{ 
		int index = getUnusedParticle();
		Particle& p = container[index]; // shortcut

		//starting speed and position
		double theta = (rand() % 628)*0.01;
		double phi = (rand() % 314)*0.01;

		p.pos = Particle::vec3(sin(phi)*cos(theta) * 10, cos(phi)*cos(theta) + 50,
			sin(phi)*sin(theta) * 10);
		p.speed = Particle::vec3(theta, -(rand() % 45 + 155), phi);
			//p.pos *-0.05;
		p.life = 5.0f; //lasts 5 seconds

		//setting misc parameters randomly for now
		p.size = ((rand() % 1000) / 2000.0f + 0.1f)*0.05;
		p.angle = (rand() % 100)*0.01;
		p.weight = (rand() % 100)*0.01;

		//start color
		p.r = 83;
		p.g = 119;
		p.b = 122;
		p.a = 255;

		//new particles added to back of camera queue
		p.cameradistance = -1.0f;

		last_particle_count = index;
	}
} 
void ParticleContainer::UpdateParticles(double delta)
{	
	int start_time = glutGet(GLUT_ELAPSED_TIME);
	int time;

	addNewParticles(delta);
	RECORD_SPEED("Add new particles %d ms \n");

	applyPhysics(delta);
	RECORD_SPEED("Apply physics  %d ms \n");


	render_counter = 0;

	//for rendering blended particles
	std::sort(container, container + max_particle_count);

	for (int i = 0; i< max_particle_count; i++){

		Particle& p = container[i]; // shortcut

		if (p.life > 0.0f){
			// Decrease life
			p.life -= delta;

			//remove particles that have drifted out of view
			if (Particle::vec3::dot(p.pos, p.pos) > 10000)
				p.life = -1;

			//change color
			int col = rand() % 5;
			int r = (p.r * 15 + colors[col][0]) / 16; p.r = (char)r;
			int g = (p.g * 15 + colors[col][1]) / 16; p.g = (char)g;
			int b = (p.b * 15 + colors[col][2]) / 16; p.b = (char)b;

			if (p.life > 0.0f){			

				// Fill the GPU buffer
				g_particle_position_data[4 * render_counter + 0] = p.pos.x;
				g_particle_position_data[4 * render_counter + 1] = p.pos.y;
				g_particle_position_data[4 * render_counter + 2] = p.pos.z;

				g_particle_position_data[4 * render_counter + 3] = p.size;

				g_particle_color_data[4 * render_counter + 0] = p.r;
				g_particle_color_data[4 * render_counter + 1] = p.g;
				g_particle_color_data[4 * render_counter + 2] = p.b;
				g_particle_color_data[4 * render_counter + 3] = p.a;

				render_counter++;
			}
			else{
				// Particles that just died will be put at the end of the buffer in SortParticles();
				p.cameradistance = -1.0f;
			}

		}
	}

	RECORD_SPEED("Copy to buffer  %d ms \n");
}
