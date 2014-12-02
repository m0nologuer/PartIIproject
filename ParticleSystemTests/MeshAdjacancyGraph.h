#pragma once
#include <assimp/scene.h>
#include <vector>
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

class MeshAdjacancyGraph
{
	//for edges of the mesh, building the hash table
	typedef std::pair<int, int> Edge;
	struct EdgeHash {
		size_t operator()(const Edge& e) const {
			//calculate hash here.
			std::hash<int> hash;
			return hash(hash(e.first) + hash(e.second));
		}
	};

	//the adjacancy graph
	struct Node{ int* adjacent_nodes; };
	Node* adjacancy_graph = NULL;

	//list of face indices for each vertex
	std::vector<int>* face_list = NULL;
	aiMesh stored_mesh;

public:
	MeshAdjacancyGraph();
	~MeshAdjacancyGraph();
	bool buildGraph(const aiMesh* mesh);
	bool vertexFanProperty();
};

