#pragma once
#include "ParticleContainer.h"

class KDTree
{
	enum Axis{ X_AXIS, Y_AXIS, Z_AXIS };
	struct Node
	{
		Particle* p;
		Node* left;
		Node* right;
		Axis ax;
	};
	class Comparator
	{
	public:
		bool operator() (Particle* a, Particle* b);
		KDTree::Axis ax;
	} comp;
	Node* root_node;
	Particle* findMedian(std::vector<Particle*> paricles, Axis a);
	Node* buildNode(std::vector<Particle*> paricles, Axis a);
	void findNeighbouringParticles(Node* n, Particle p, 
		std::vector<Particle*>& list, double rad);
public:
	KDTree(Particle* particle_container, int particle_count);
	~KDTree();
	std::vector<Particle*> findNeighbouringParticles(Particle p, double rad);
};

