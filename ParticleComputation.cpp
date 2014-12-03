#include "ParticleContainer.h"
#include "ParticleKernels.h"

#include <stdio.h>
#include "KDTree.h"

#define USE_KDTREE 0

void ParticleContainer::applyPhysics(double delta)
{
	delta = delta * 0.001;

#ifdef USE_KDTREE
	//start by building the k-D tree
	KDTree tree(container, max_particle_count);
#endif 

	//update data structure
	for (int i = 0; i < max_particle_count; i++){
		Particle& p = container[i]; // shortcut

		//if particle is alive
		if (p.life > 0.0f){

			//update  speed and position
			p.speed = p.speed + getParticleForce(p.pos) * (double)delta;
			p.predicted_pos = p.pos + p.speed * (double)delta;

			//find neighbouring particles
#ifdef USE_KDTREE
			neighbours[i] = tree.findNeighbouringParticles(p, h);
#else
			neighbours[i] = findNeighbouringParticles(p);
#endif
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
	vector<Particle*>::iterator it;
	Particle::vec3 sum(0, 0, 0);
	for (it = neighbours[index].begin(); it < neighbours[index].end(); it++) {
		Particle pj = (**it);

		//calculate correction
		double s_corr = W_poly6(pi.predicted_pos - pj.predicted_pos) / Wq;
		s_corr = -corr_k * pow(s_corr, n);

		//add contribution
		sum = sum + dW_spiky(pi.predicted_pos - pj.predicted_pos)*(pi.lambda + pj.lambda + s_corr);
	}
	//scaled sum
	return sum *(1 / p0);
}
double ParticleContainer::constraint_function(int index){

	Particle pi = container[index];

	//iterate over all neighbours
	vector<Particle*>::iterator it;
	double sum = 0;
	for (it = neighbours[index].begin(); it < neighbours[index].end(); it++) {
		Particle pj = (**it);
		//smoothing kernel
		sum += W_poly6(pi.predicted_pos - pj.predicted_pos);
	}

	double Ci = sum / p0 - 1;

	return Ci;
}
double ParticleContainer::gradient_constraint_function(int i, Particle* pk){
	Particle* pi = &container[i];
	if (pi == pk) //differentiating the above function
	{
		vector<Particle*>::iterator it;
		double sum = 0;
		for (it = neighbours[i].begin(); it < neighbours[i].end(); it++) {
			Particle pj = (**it);
			//taking gradient of scalar field, dot with d(pk)/dt
			sum += Particle::vec3::dot(dW_poly6(pi->predicted_pos - pj.predicted_pos), pj.speed);
		}
		return sum / p0;
	}
	else
	{
		Particle pj = *pk;
		double result = -Particle::vec3::dot(dW_poly6(pi->predicted_pos - pj.predicted_pos), pj.speed);

		return result / p0;
	}
}
double ParticleContainer::lambda(int index)
{
	double numerator = constraint_function(index);

	vector<Particle*>::iterator it;
	double denomninator = e0;
	for (it = neighbours[index].begin(); it < neighbours[index].end(); it++) {
		denomninator += pow(gradient_constraint_function(index, (*it)), 2.0);
	}

	return -numerator / denomninator;
}

Particle::vec3 ParticleContainer::getParticleForce(Particle::vec3 pos)
{
	// Simulate simple physics : gravity only, no collisions
	return Particle::vec3(0, -9.81, 0);
}
vector<Particle*> ParticleContainer::findNeighbouringParticles(Particle postion)
{
	vector<Particle*> neighbours;
	for (int i = 0; i < max_particle_count; i++){
		if (container[i].life > 0.0f){
			Particle::vec3 direction = postion.pos - container[i].pos;
			double distance = Particle::vec3::dot(direction, direction);
			if (distance < h_squared)
				neighbours.push_back(&(container[i]));
		}
	}
	return neighbours;
}