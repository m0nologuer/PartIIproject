#include "stdafx.h"
#include "MeshAdjacancyGraph.h"
#include <unordered_map>
#include <deque>

bool MeshAdjacancyGraph::vertexFanProperty()
{
	//check the vertex fan has been built
	if (!(adjacancy_graph && face_list))
		return false;

	//check every vertex
	for (int i = 0; i < stored_mesh.mNumVertices; i++)
	{
		//check no isolated vertices
		bool vertex_attached = face_list[i].empty();
		Assert::IsTrue(vertex_attached);
		if (!vertex_attached) return false;

		//check list of faces form a fan
		int face_fan_size = face_list[i].size();
		int first_face_index = face_list[i][0];
		int current_face = first_face_index;
		int prev_face = first_face_index;
		int face_counter = 0;
		//traverse the vertex's neighbouring faces in a loop 
		do{
			Node face_node = adjacancy_graph[current_face];
			aiFace face = stored_mesh.mFaces[current_face];
			for (int v = 0; v < face.mNumIndices; v++)
				if (face.mIndices[v] == i) //find the vertex i
				{
					int v_j = v < (face.mNumIndices - 1) ? v + 1 : 0;
					int adj_face = face_node.adjacent_nodes[v]; //next face in fan

					if (adj_face == prev_face) //make sure we're not going backwards
						adj_face = face_node.adjacent_nodes[v_j];
					
					//make sure we're not a degenerate face
					Assert::IsTrue(adj_face != current_face);

					//update face
					prev_face = current_face;
					current_face = adj_face;
				}
			//check we aren't looping around too many triangles
			face_counter++;
			Assert::IsTrue(face_fan_size > face_counter);
			
		} while (current_face != first_face_index);

		//check we looped around the right number of times
		Assert::IsTrue(face_fan_size == face_counter);

	}
	//otherwise we have found successful 'fans' for every vertex
	return true;
}

bool MeshAdjacancyGraph::buildGraph(const aiMesh* mesh)
{
	//Start by building a hash table mapping clockwise edges to face indices
	std::unordered_map<Edge, int, EdgeHash> hash_table;
	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		for (int j = 0; j < face.mNumIndices; j++)
		{
			//clockwise edge face
			int n_j = j < (face.mNumIndices - 1) ? j + 1 : 0;
			Edge new_edge = std::make_pair((int)face.mIndices[j],
				(int)face.mIndices[n_j]);

			//each ordered edge should only occur once
			bool repeated_edge = (hash_table.count(new_edge) > 0);
			Assert::IsFalse(repeated_edge);
			if (repeated_edge) return false;

			//add to hash
			hash_table.insert(std::make_pair(new_edge, i));
		}
	}

	//initialize adj graph
	adjacancy_graph = new Node[mesh->mNumFaces];
	for (int i = 0; i < mesh->mNumFaces; i++)
		adjacancy_graph[i].adjacent_nodes = NULL;

	//deque for trversing graph
	std::deque<int> face_queue;
	face_queue.push_back(0);
	int face_counter = 0;

	while (!face_queue.empty())
	{
		//take current face
		int face_index = face_queue.front();
		face_queue.pop_front();
		aiFace face = mesh->mFaces[face_index];

		//skip if we've already initialized
		if (adjacancy_graph[face_index].adjacent_nodes != NULL)
			continue;

		//initialize node
		adjacancy_graph[face_index].adjacent_nodes = new int[face.mNumIndices];
		face_counter++;

		//Match each edge with one of an opposite orientation
		for (int j = 0; j < face.mNumIndices; j++)
		{
			//counterclockwise edge face
			int n_j = j < (face.mNumIndices - 1) ? j + 1 : 0;
			Edge new_edge = std::make_pair((int)face.mIndices[n_j],
				(int)face.mIndices[j]);

			//the opposite orientation edge should exist
			bool opposite_edge = (hash_table.count(new_edge) == 0);
			Assert::IsTrue(opposite_edge);
			if (!opposite_edge) return false;

			//retrive index of the face its attached to
			int other_edge = hash_table[new_edge];
			adjacancy_graph[face_index].adjacent_nodes[j] = other_edge;
			//add to queue	
			face_queue.push_back(other_edge);
		}
	}

	//if the mesh is not connected, and we have not traversed every face
	bool traversed_mesh = !(face_counter < mesh->mNumFaces);
	Assert::IsTrue(traversed_mesh);
	if (!traversed_mesh) return false;

	//otherwise, we have successfully built the graph
	//now build an index of faces to vertices
	stored_mesh = *mesh;
	face_list = new std::vector<int>[mesh->mNumVertices];
	for (int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		for (int j = 0; j < face.mNumIndices; j++)
		{
			int index = face.mIndices[j];
			face_list[index].push_back(j);
		}
	}
	return true;
}

MeshAdjacancyGraph::MeshAdjacancyGraph()
{
}


MeshAdjacancyGraph::~MeshAdjacancyGraph()
{
	delete[] adjacancy_graph;
	delete[] face_list;
}
