
#include <stdlib.h>
#include <stdio.h>
#include <GL/glut.h>

// assimp include files. These three are usually needed.
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <vector>

#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)

class ColladaLoader{

public:
	ColladaLoader();
	~ColladaLoader();
	void render();
	float* data(int& n);
	bool inside_mesh(aiVector3D position);
	int loadasset(const char* path);
	const aiScene* getSceneObject();

	// current rotation angle
	float angle = 0.f;
	aiVector3D scene_min, scene_max, scene_center;
private:
	// the global Assimp scene object
	const aiScene* scene = NULL;

	void get_bounding_box_for_node(const aiNode* nd,
		aiVector3D* min,
		aiVector3D* max,
		aiMatrix4x4* trafo
		);
	void get_bounding_box(aiVector3D* min, aiVector3D* max);
	void recursive_render(const aiScene *sc, const aiNode* nd);
	bool recursive_inside_mesh(const aiScene *sc, const aiNode* nd,
		aiVector3D position);
	void recursive_data(const aiScene *sc, const aiNode* nd, aiMatrix4x4 mat, std::vector<aiVector3D>& vertices);
	void applyMatrix(const aiScene *sc, const aiNode* nd, aiMatrix4x4 mat);
};