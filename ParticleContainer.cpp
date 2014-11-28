#include "ParticleContainer.h"
#include "ParticleContainerGL.h"

#include <stdio.h>

ParticleContainer::ParticleContainer()
{
	last_particle_count = 0;
	for (int i = 0; i < max_particle_count; i++)
	{
		container[i].life = -1; //initalize all particles dead
	}
	average_speed = 0;
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
		double phi = 1.7;

		p.pos = Particle::vec3(sin(phi)*cos(theta), cos(phi)+5, sin(phi)*sin(theta))*0.1;
		p.speed = Particle::vec3(rand() % 10 - 5, 100 + rand() % 50, rand() % 10 - 5)*-1;
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
void ParticleContainer::applyPhysics(double delta)
{
	delta = delta * 0.001;

	//update data structure
	for (int i = 0; i < max_particle_count; i++){
		Particle& p = container[i]; // shortcut

		//if particle is alive
		if (p.life > 0.0f){

			//update  speed and position
			p.speed = p.speed + getParticleForce(p.pos) * (double)delta;
			p.predicted_pos = p.pos + p.speed * (double)delta;
			
			//find neighbouring particles
			neighbours[i] = findNeighbouringParticles(p);
		}
	}

	//Perform collision detection, solving
	for (int i = 0; i < iteration_count; i++)
		solverIteration();

	//Update particle info
	for (int i = 0; i < max_particle_count; i++){
		container[i].speed = (container[i].predicted_pos - container[i].pos * 1.0) *(1 / delta);
		container[i].pos = container[i].predicted_pos;
	}
}
void ParticleContainer::solverIteration()
{
	//calculate lambdas
	for (int i = 0; i < max_particle_count; i++){
		Particle& p = container[i]; // shortcut
		if (p.life > 0.0f){
			p.lambda = lambda(i);
		}
	}
	
	//perform collision detection & adjustment
	for (int i = 0; i < max_particle_count; i++){
		Particle& p = container[i]; // shortcut
		if (p.life > 0.0f){
			p.d_p_pos = collisionUpdate(i);
		}
	}

	for (int i = 0; i < max_particle_count; i++){
		container[i].predicted_pos = container[i].predicted_pos + container[i].d_p_pos;
	}

}
Particle::vec3 ParticleContainer::collisionUpdate(int index)
{
	Particle pi = container[index];

	//iterate over neighbours
	vector<int>::iterator it;
	Particle::vec3 sum(0,0,0);
	for (it = neighbours[index].begin(); it < neighbours[index].end(); it++) {
		Particle pj = container[(*it)];
		sum = sum + dW_poly6(pi.predicted_pos - pj.predicted_pos)*(pi.lambda+pj.lambda);
	}
	//scaled sum
	return sum *(1 / p0);
}
double ParticleContainer::constraint_function(int index){

	Particle pi = container[index];
	
	//iterate over all neighbours
	vector<int>::iterator it;
	double sum = 0;
	for (it = neighbours[index].begin(); it < neighbours[index].end(); it++) {
		Particle pj = container[(*it)];
		//smoothing kernel
		sum += W_poly6(pi.predicted_pos - pj.predicted_pos);
	}

	double Ci = sum / p0 - 1;

	return Ci;
}
double ParticleContainer::gradient_constraint_function(int i, int k){
	Particle pi = container[i];

	if (i == k) //differentiating the above function
	{ 
		vector<int>::iterator it;
		double sum = 0;
		for (it = neighbours[i].begin(); it < neighbours[i].end(); it++) {
			Particle pj = container[(*it)];
			//taking gradient of scalar field, dot with d(pk)/dt
			sum += Particle::vec3::dot(dW_poly6(pi.predicted_pos - pj.predicted_pos), pj.speed);
		}
		return sum / p0;
	}
	else
	{
		Particle pj = container[k];
		double result = -Particle::vec3::dot(dW_poly6(pi.predicted_pos - pj.predicted_pos), pj.speed);

		return result / p0;
	}
}
double ParticleContainer::lambda(int index)
{
	Particle pi = container[index];

	double numerator = constraint_function(index);

	vector<int>::iterator it;
	double denomninator = e0;
	for (it = neighbours[index].begin(); it < neighbours[index].end(); it++) {
		denomninator += pow(gradient_constraint_function(index, (*it)), 2.0);
	}

	return -numerator / denomninator;
}
double ParticleContainer::W_spiky(Particle::vec3 r)
{
	return 0;
}
double ParticleContainer::W_poly6(Particle::vec3 r){

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
Particle::vec3 ParticleContainer::dW_spiky(Particle::vec3 r){

	return Particle::vec3(0,0,0);
}
Particle::vec3 ParticleContainer::dW_poly6(Particle::vec3 r){

	double radius_2 = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius_2 < h_squared)
	{
		//constant is 315/64pi
		double radius = sqrt(radius_2);
		double result = -6 * pow(h_squared - radius_2, 2.0) * 1.56668147 / h_9;
		Particle::vec3 grad = r * result;
		return grad;
	}
	else //ignore particles outside a certain large radius
		return Particle::vec3(0,0,0);
}

Particle::vec3 ParticleContainer::getParticleForce(Particle::vec3 pos)
{ 
	// Simulate simple physics : gravity only, no collisions
	if (pos.x > -0.05)
		return Particle::vec3(0, -9.81, 0);
	else
		return Particle::vec3(0, 900.81, 0);
}
vector<int> ParticleContainer::findNeighbouringParticles(Particle postion)
{
	vector<int> neighbours;
	for (int i = 0; i < max_particle_count; i++){
		if (container[i].life > 0.0f){
			Particle::vec3 direction = postion.pos - container[i].pos;
			double distance = Particle::vec3::dot(direction, direction);
			if (distance < h_squared)
				neighbours.push_back(i);
		}
	}
	return neighbours;
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
