#include "COLLADALoader.h"
#include <math.h>

ColladaLoader::ColladaLoader(){

}
ColladaLoader::~ColladaLoader(){
	aiReleaseImport(scene);

}
const aiScene* ColladaLoader::getSceneObject()
{
	return scene;
}

// ----------------------------------------------------------------------------
void ColladaLoader::get_bounding_box_for_node(const aiNode* nd,
	aiVector3D* min,
	aiVector3D* max,
	aiMatrix4x4* trafo
	){
	aiMatrix4x4 prev;
	unsigned int n = 0, t;

	prev = *trafo;
	aiMultiplyMatrix4(trafo, &nd->mTransformation);

	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t) {

			aiVector3D tmp = mesh->mVertices[t];
			aiTransformVecByMatrix4(&tmp, trafo);

			min->x = aisgl_min(min->x, tmp.x);
			min->y = aisgl_min(min->y, tmp.y);
			min->z = aisgl_min(min->z, tmp.z);

			max->x = aisgl_max(max->x, tmp.x);
			max->y = aisgl_max(max->y, tmp.y);
			max->z = aisgl_max(max->z, tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		get_bounding_box_for_node(nd->mChildren[n], min, max, trafo);
	}
	*trafo = prev;
}

// ----------------------------------------------------------------------------
void ColladaLoader::get_bounding_box(aiVector3D* min, aiVector3D* max)
{
	aiMatrix4x4 trafo;
	aiIdentityMatrix4(&trafo);

	min->x = min->y = min->z = 1e10f;
	max->x = max->y = max->z = -1e10f;
	get_bounding_box_for_node(scene->mRootNode, min, max, &trafo);
}

// ----------------------------------------------------------------------------

bool ColladaLoader::inside_mesh(aiVector3D position)
{
	return recursive_inside_mesh(scene, scene->mRootNode, position);
}
bool ColladaLoader::recursive_inside_mesh(const aiScene *sc, const aiNode* nd, aiVector3D position)
{
	aiMatrix4x4 mat = nd->mTransformation;
	mat = mat.Inverse();
	position = mat * position;


	int intersect_count = 0;

	for (int n = 0; n < nd->mNumMeshes; ++n) {
	  	//find meshes and for each face...
		const aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
		{
			for (int t = 0; t < mesh->mNumFaces; ++t) {
				const aiFace* face = &mesh->mFaces[t];
				//if this is a real face
				if (face->mNumIndices > 2)
				{
					//raycast along line pararell to z axis
					aiVector3D dir = position - mesh->mVertices[face->mIndices[0]];
					for (int i = 1; i < face->mNumIndices - 1; i++)
					{
						aiVector3D axis1 = mesh->mVertices[face->mIndices[i]]
							- mesh->mVertices[face->mIndices[0]];
						aiVector3D axis2 = mesh->mVertices[face->mIndices[i + 1]]
							- mesh->mVertices[face->mIndices[0]];

						double l1 = axis1*dir;
						double l2 = axis2*dir;

						//if there is an intersection 
						if (l1 >= 0 && l1 <= 1 && l2 >= 0 && l2 <= 1 && l1 + l2 <= 1)
							intersect_count++;
					}
				}
			} 
		}
	}

	//if we get an odd number of intersections, we're inside this mesh
	if (intersect_count % 2 == 1)
		return true;

	//check the other meshes
	bool intersection = false;
	for (int n = 0; n < nd->mNumChildren; ++n) {
		intersection = intersection || recursive_inside_mesh(sc, nd->mChildren[n], position);
	}
	  
	return intersection;
}
float* ColladaLoader::data(int& block_size){

	std::vector<aiVector3D> vertices;

	aiMatrix4x4 identity = aiMatrix4x4(); 
	recursive_data(scene, scene->mRootNode, identity, vertices);
	block_size = vertices.size()/3;

	float* data_blob = new float[4 * vertices.size()];
	float* p_data = data_blob;

	for (int i = 0; i < vertices.size(); i+=3)
	{
		float* pos = data_blob;
		float* normal = data_blob + 3;
		float* pos_a = data_blob + 6;
		float* pos_b = data_blob + 9;

		pos[0] = vertices[i].x;
		pos[1] = vertices[i].y;
		pos[2] = vertices[i].z;

		aiVector3D dir1_v = vertices[i + 2] - vertices[i];
		aiVector3D dir2_v = vertices[i + 1] - vertices[i];
		aiVector3D normal_v = dir1_v.SymMul(dir2_v);
		normal_v.Normalize();
		normal[0] = normal_v.x;
		normal[1] = normal_v.y;
		normal[2] = normal_v.z;

		pos_a[0] = vertices[i + 1].x;
		pos_a[1] = vertices[i + 1].y;
		pos_a[2] = vertices[i + 1].z;

		pos_b[0] = vertices[i + 2].x;
		pos_b[1] = vertices[i + 2].y;
		pos_b[2] = vertices[i + 2].z;


		/*
		dir1_v *= 1.0f / dir1_v.SquareLength();
		dir2_v *= 1.0f / dir2_v.SquareLength();

		dir1[0] = dir1_v.x;
		dir1[1] = dir1_v.y;
		dir1[2] = dir1_v.z;

		dir2[0] = dir2_v.x;
		dir2[1] = dir2_v.y;
		dir2[2] = dir2_v.z;
		*/

		data_blob += 12;
	}

	return p_data;
}
void ColladaLoader::applyMatrix(const aiScene *sc, const aiNode* nd, aiMatrix4x4 mat)
{
	unsigned int i;
	unsigned int n = 0, t;
	aiMatrix4x4 m = nd->mTransformation;
	mat *= m;

	// draw all meshes assigned to this node
	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];
		for (i = 0; i < mesh->mNumVertices; i++) {
			mesh->mVertices[i] *= mat;
		}
	}

	// draw all children
	for (n = 0; n < nd->mNumChildren; ++n) {
		applyMatrix(sc, nd->mChildren[n], mat);
	}
}
void ColladaLoader::recursive_data(const aiScene *sc, const aiNode* nd, aiMatrix4x4 mat, std::vector<aiVector3D>& vertices)
{
	unsigned int i;
	unsigned int n = 0, t;
	aiMatrix4x4 m = nd->mTransformation;
	mat *= m;

	// draw all meshes assigned to this node
	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]]; 
		for (t = 0; t < mesh->mNumFaces; ++t) {
			const aiFace* face = &mesh->mFaces[t];
			if (face->mNumIndices != 3)
				break;
			for (i = 0; i < face->mNumIndices; i++) {
				int index = face->mIndices[i];
				aiVector3D vec = mesh->mVertices[index];
				//vec *= mat;
				vertices.push_back(vec);
			}
		}
	}

	// draw all children
	for (n = 0; n < nd->mNumChildren; ++n) {
		recursive_data(sc, nd->mChildren[n], mat, vertices);
	}
}
// ----------------------------------------------------------------------------
void ColladaLoader::render(){
	recursive_render(scene, scene->mRootNode);
}
void ColladaLoader::recursive_render(const aiScene *sc, const aiNode* nd)
{
	unsigned int i;
	unsigned int n = 0, t;
	aiMatrix4x4 m = nd->mTransformation;

	// update transform
	aiTransposeMatrix4(&m);
	glPushMatrix();
	glMultMatrixf((float*)&m);

	// draw all meshes assigned to this node
	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

		if (mesh->mNormals == NULL) {
			glDisable(GL_LIGHTING);
		}
		else {
			glEnable(GL_LIGHTING);
		}

		for (t = 0; t < mesh->mNumFaces; ++t) {
			const aiFace* face = &mesh->mFaces[t];
			GLenum face_mode;

			switch (face->mNumIndices) {
			case 1: face_mode = GL_POINTS; break;
			case 2: face_mode = GL_LINES; break;
			case 3: face_mode = GL_TRIANGLES; break;
			default: face_mode = GL_POLYGON; break;
			}

			glBegin(face_mode);

			for (i = 0; i < face->mNumIndices; i++) {
				int index = face->mIndices[i];
				if (mesh->mColors[0] != NULL)
					glColor4fv((GLfloat*)&mesh->mColors[0][index]);
				if (mesh->mNormals != NULL)
					glNormal3fv(&mesh->mNormals[index].x);
				glVertex3fv(&mesh->mVertices[index].x);
			}

			glEnd();
		}

	}

	// draw all children
	for (n = 0; n < nd->mNumChildren; ++n) {
		recursive_render(sc, nd->mChildren[n]);
	}

	glPopMatrix();
}


// ----------------------------------------------------------------------------
int ColladaLoader::loadasset(const char* path)
{
	// we are taking one of the postprocessing presets to avoid
	// spelling out 20+ single postprocessing flags here.
	scene = aiImportFile(path, aiProcessPreset_TargetRealtime_MaxQuality);

	if (scene) {
		get_bounding_box(&scene_min, &scene_max);
		scene_center.x = (scene_min.x + scene_max.x) / 2.0f;
		scene_center.y = (scene_min.y + scene_max.y) / 2.0f;
		scene_center.z = (scene_min.z + scene_max.z) / 2.0f;

		//transform
		float tmp = scene_max.x - scene_min.x;
		tmp = std::max(scene_max.y - scene_min.y, tmp);
		tmp = std::max(scene_max.z - scene_min.z, tmp);
		tmp = 40.0f/tmp;
		aiMatrix4x4 matrix = aiMatrix4x4(aiVector3D(tmp, tmp, tmp), aiQuaternion(), aiVector3D(0,0,0));
		matrix *= aiMatrix4x4(aiVector3D(1, 1, 4), aiQuaternion(aiVector3D(0,0,1),3.1416*0.25), -scene_center/tmp + aiVector3D(0,-20,-15)/tmp);
		applyMatrix(scene, scene->mRootNode, matrix);

		return 0;
	}
	return 1;
}