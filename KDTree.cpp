#include "KDTree.h"
#include <algorithm>

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
KDTree::KDTree(Particle* particle_container, int particle_count)
{
	//only count alive particles
	std::vector<Particle*> particles;
	for (int i = 0; i < particle_count; i++)
		if (particle_container[i].life > 0)
			particles.push_back(&(particle_container[i]));
	
	//unrolled recursive process to stop stack overflow
	root_node = new KDTree::Node();
	KDTreeArgs tree_args(&particles, Axis::X_AXIS, root_node);
	build_deque.push_back(tree_args);
	//build the tree
	while (!build_deque.empty())
	{
		KDTreeArgs next_tree_args = build_deque.front();
		buildNode(next_tree_args.particles, next_tree_args.a, next_tree_args.node);
		build_deque.pop_front();
	}

}
Particle* KDTree::findMedian(std::vector<Particle*> particles, KDTree::Axis a)
{
	int half_n = particles.size() / 2;
	comp.ax = a;
	std::nth_element(particles.begin(), particles.begin() + half_n, particles.end(),
		comp);
	return particles[half_n];
}
void KDTree::buildNode(std::vector<Particle*>* particles, KDTree::Axis a, KDTree::Node* node)
{
	//create new node
	Particle* median_p = findMedian(*particles, a); //find hyperplane
	node->p = median_p;
	node->left = NULL;
	node->right = NULL;
	node->ax = a;
	comp.ax = a; //set compatator

	//sort particles across hyperplane
	std::vector<Particle*>* left = new std::vector<Particle*>();
	std::vector<Particle*>* right = new std::vector<Particle*>();
	
	for (size_t i = 0; i < particles->size(); i++)
		if ((*particles)[i] != median_p) //make sure to not sort the median element
		{
			if (comp((*particles)[i], median_p))
				right->push_back((*particles)[i]);
			else
				left->push_back((*particles)[i]);
		}

	//recursively build nodes
	KDTree::Axis next_a = (KDTree::Axis)(((int)a + 1) % 3);
	if (!left->empty())
	{
		node->left = new KDTree::Node();
		KDTreeArgs tree_args(left, next_a, node->left);
		build_deque.push_back(tree_args);
	}
	if (!right->empty())
	{
		node->right = new KDTree::Node();
		KDTreeArgs tree_args(right, next_a, node->right);
		build_deque.push_back(tree_args);
	}
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
			traversal_deque.push_back(n->left);
		if (n->right)
			traversal_deque.push_back(n->right);
	}
	else
	{
		//see which side we need to look at
		comp.ax = n->ax;
		bool right = comp(n->p, &p);

		//recursively check the structure for the right particles
		if (!right && n->left)
			traversal_deque.push_back(n->left);
		if (right && n->right)
			traversal_deque.push_back(n->right);

	}
}
std::vector<Particle*> KDTree::findNeighbouringParticles(Particle p, double rad)
{
	std::vector<Particle*> empty_list;
	if (!(root_node && traversal_deque.empty())) //made sure tree is built and deque free
		return empty_list;
	
	rad = rad*rad; //to save squaing every time

	//unrolled for stack reasons
	traversal_deque.push_back(root_node);
	while (!traversal_deque.empty())
	{
		Node* next_node = traversal_deque.front();
		KDTree::findNeighbouringParticles(next_node, p, empty_list, rad);
		traversal_deque.pop_front();
	}
	return empty_list;
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

			if (current_node)
			{
			///	delete current_node;
				if (current_node->left)
					active_nodes.push_back(current_node->left);
				if (current_node->right)
					active_nodes.push_back(current_node->left);
				current_node = NULL;
			}
			active_nodes.pop_front();
		}
	}
}
