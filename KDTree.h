#pragma once
#include "Particle.h"
#include "PerformanceSettings.h"
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
#ifdef USE_CUDA
		int *particle_blob;
		int blob_size;
#endif
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
	std::vector<Node*> all_nodes;
	Particle* container; //zeroth entry for current tree

	Particle* findMedian(std::vector<Particle*> paricles, Axis a);
	void buildNode(std::vector<Particle*>* particles, Axis a, Node* node);
	void findNeighbouringParticles(Node* n, Particle p, 
		std::vector<Particle*>& list, double rad);

#ifdef USE_CUDA
	//How to excecute the kernel
	int particle_blob_size;

	//initializing
	void initialize_CUDA();
	void initialize_CUDA_node(Node* p);
	void destroy_CUDA();
	void destroy_CUDA_node(Node* p);

	//finding the correct particle
	void batchNeighbouringParticles_CUDA(const Node* n, double rad,
		 float* particles_CUDA, int* neighbours_CUDA);
#endif

public:
	KDTree(Particle* particle_container, int particle_count);
	~KDTree();
	std::vector<Particle*> findNeighbouringParticles(Particle p, double rad);
#ifdef USE_CUDA
	static KDTree* treeFromFloats(float* particle_positions, int particle_count);
	void batchNeighbouringParticles(double rad, float* particle_pos_CUDA, int* neighbours_CUDA);
#endif
};

