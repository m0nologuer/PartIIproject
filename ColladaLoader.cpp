#include "COLLADALoader.h"

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
		return 0;
	}
	return 1;
}