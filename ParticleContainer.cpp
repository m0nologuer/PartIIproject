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

	Wq = W_poly6(Particle::vec3(q,0,0));
	h = 0.1; //distance paramter
	e0 = 0.01; //relaxion parameter

	//for the correction 
	corr_k = 0.1;
	q = 0.1*h;
	p0 = 0.5; //rest desngity
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
	int new_particles = delta*particles_per_second > particles_per_second ? particles_per_second : (int)(delta * particles_per_second); //100 particles per second
	new_particles++;

	for (int i = 0; i < new_particles; i++)
	{
		int index = getUnusedParticle();
		Particle& p = container[index]; // shortcut

		//starting speed and position
		double theta = (rand() % 628)*0.01;
		double phi = (rand() % 314)*0.01;

		p.pos = Particle::vec3(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta))*0.1;
		p.speed = Particle::vec3(0,1,0)*50;
		p.life = 5.0f; //lasts 5 seconds

		//setting misc parameters randomly for now
		p.size = ((rand() % 1000) / 2000.0f + 0.1f)*0.05;
		p.angle = (rand() % 100)*0.01;
		p.weight = (rand() % 100)*0.01;

		//random color
		p.r = rand() % 255;
		p.g = rand() % 255;
		p.b = rand() % 255;
		p.a = rand() % 255;

		//new particles added to back of camera queue
		p.cameradistance = -1.0f;

		last_particle_count = index;
	}
} 
void ParticleContainer::UpdateParticles(double delta)
{	
	addNewParticles(delta); 
	applyPhysics(delta);

	render_counter = 0;

	for (int i = 0; i< max_particle_count; i++){

		Particle& p = container[i]; // shortcut

		if (p.life > 0.0f){
			// Decrease life
			p.life -= delta;

			if (p.life > 0.0f){			

				//For particle sorting
				Particle::vec3 c_vec = p.pos + (camera_pos * -1.0f);
				p.cameradistance = sqrt(c_vec.x*c_vec.x + c_vec.y*c_vec.y + c_vec.z*c_vec.z);

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
}
