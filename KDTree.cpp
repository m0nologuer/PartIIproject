#include "KDTree.h"
#include <algorithm>
#include <deque>

bool KDTree::Comparator::operator() (Particle* a, Particle* b)
{
	bool greater;
	switch (ax)
	{
	case X_AXIS:
		greater = (a->pos.x > b->pos.x);
	case Y_AXIS:
		greater = (a->pos.y > b->pos.y);
	case Z_AXIS:
		greater = (a->pos.z > b->pos.z);
	}
	return greater;
}

Particle* KDTree::findMedian(std::vector<Particle*> particles, KDTree::Axis a)
{
	int half_n = particles.size() / 2;
	comp.ax = a;
	std::nth_element(particles.begin(), particles.begin() + half_n, particles.end(),
		comp);
	return particles[half_n];
}
KDTree::Node* KDTree::buildNode(std::vector<Particle*> particles, KDTree::Axis a)
{
	if (particles.empty())
		return NULL;

	//create new node
	KDTree::Node* node = new KDTree::Node;
	Particle* median_p = findMedian(particles, a); //find hyperplane
	node->p = median_p;
	node->left = NULL;
	node->right = NULL;
	node->ax = a;
	comp.ax = a; //set compatator

	//sort particles across hyperplane
	std::vector<Particle*> left, right;
	for (size_t i = 0; i < particles.size(); i++)
	{
		if (comp(particles[i], median_p))
			right.push_back(particles[i]);
		else
			left.push_back(particles[i]);
	}

	//recursively build nodes
	KDTree::Axis next_a = (KDTree::Axis)(((int)a + 1) % 3);
	if (!left.empty())
		node->left = buildNode(left, next_a);
	if (!right.empty())
		node->right = buildNode(right, next_a);

	return node;
}
void KDTree::findNeighbouringParticles(Node* n, Particle p, 
	std::vector<Particle*>& list, double rad)
{
	Particle::vec3 dist = p.pos - n->p->pos;
	double distance_sq = Particle::vec3::dot(dist, dist);
	if (distance_sq < rad)
	{ 
		//if this node is close enough, add it to the list
		list.push_back(n->p);
		//then consider nodes on both sides
		if (n->left)
			findNeighbouringParticles(n->left, p, list, rad);
		if (n->right)
			findNeighbouringParticles(n->right, p, list, rad);
	}
	else
	{
		//see which side we need to look at
		comp.ax = n->ax;
		bool right = comp(n->p, &p);

		//recursively check the structure for the right particles
		if (!right && n->left)
			findNeighbouringParticles(n->left, p, list, rad);
		if (right && n->right)
			findNeighbouringParticles(n->right, p, list, rad);

	}
}
std::vector<Particle*> KDTree::findNeighbouringParticles(Particle p, double rad)
{
	std::vector<Particle*> empty_list;
	if (root_node)
		KDTree::findNeighbouringParticles(root_node, p, empty_list, rad);
	return empty_list;
}
KDTree::KDTree(Particle* particle_container, int particle_count)
{
	std::vector<Particle*> particles;
	root_node = buildNode(particles, Axis::X_AXIS);
}
KDTree::~KDTree()
{
	//disassemble tree
	if (root_node)
	{
		std::deque<Node*> active_nodes;
		active_nodes.push_back(root_node);

		while (!active_nodes.empty())
		{
			Node* current_node = active_nodes.front();

			if (current_node->left)
				active_nodes.push_back(current_node->left);
			if (current_node->right)
				active_nodes.push_back(current_node->left);
			
			active_nodes.pop_front();
			delete current_node;
			current_node = NULL;
		}
	}
}
