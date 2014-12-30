#pragma once
#include "Particle.h"
#include <deque>
#include <math.h>
#include <cstdlib>
#include <vector>

using namespace std;

#include <GL/glut.h>
#include <GL/glext.h>

class KDTree
{
	enum Axis{ X_AXIS, Y_AXIS, Z_AXIS };
	struct Node
	{
		Particle* p;
		Node* left;
		Node* right;
		Axis ax;
		Particle *particle_blob;
		int blob_size;
	};

	//used for building the tree
	class KDTreeArgs
	{
	public:
		std::vector<Particle*>* particles;
		KDTree::Axis a;
		KDTree::Node* node;
		KDTreeArgs(std::vector<Particle*>* p, Axis ax, Node* n) 
		{
			particles = p;
			node = n;
			a = ax;
		}
	};
	std::deque<KDTreeArgs> build_deque;
	std::deque<Node*> traversal_deque;
	class Comparator
	{
	public:
		bool operator() (Particle* a, Particle* b);
		KDTree::Axis ax;
	} comp;

	Node* root_node;
	Particle* findMedian(std::vector<Particle*> paricles, Axis a);
	void buildNode(std::vector<Particle*>* particles, Axis a, Node* node);
	void findNeighbouringParticles(Node* n, Particle p, 
		std::vector<Particle*>& list, double rad);

	//Hardware accelerated functions
	bool hardware_acceleration;
	int thread_count;
	int particle_blob_size;
	bool *output_grid;
	bool *output_grid_CUDA;
	void initialize_CUDA();
	void initialize_CUDA_node(Node* p);
	void destroy_CUDA();
	void destroy_CUDA_node(Node* p);
	void findNeighbouringParticles_CUDA(Node* n, Particle p,
		std::vector<Particle*>& list, double rad);

public:
	KDTree(Particle* particle_container, int particle_count, int threads);
	~KDTree();
	std::vector<Particle*> findNeighbouringParticles(Particle p, double rad);
};

