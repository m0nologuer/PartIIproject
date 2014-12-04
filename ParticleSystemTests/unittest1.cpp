#include "stdafx.h"
#include <gl/glew.h>
#include "MeshAdjacancyGraph.h"
#include "GlobalSettings.h"
#include "CppUnitTest.h"
#include <GL/glut.h>
#include <GL/glext.h>
#include "ParticleContainer.h"
#include "TestUtil.h"
#include "KDTree.h"
#include <chrono>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ParticleSystemTests
{	
	TEST_CLASS(UnitTest1)
	{
	public:
		TEST_METHOD(GenerateGifWithSettings)
		{
			ColladaLoader c_loader;
			GlobalSettings settings;
			ParticleContainer p_container;

			//set up gl
			glutInitWindowSize(1024, 768);
			glutInitWindowPosition(100, 100);
			glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
			char* argv = "dummy";
			int argc = 1;
			glutInit(&argc, &argv);
			glutCreateWindow("TestWindow");
			Assert::IsTrue((glewInit() == 0));

			glClearColor(0.1f, 0.1f, 0.1f, 1.f);
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);    // Uses default lighting parameters
			glEnable(GL_DEPTH_TEST);
			glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
			glEnable(GL_NORMALIZE);
			glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
			glutGet(GLUT_ELAPSED_TIME);

			//Load 3D model
			Assert::IsTrue(settings.LoadFromJson("../../assets/settings.json"));
			char* model_name = settings.getAssetLocation("model_name");
			c_loader.loadasset(model_name);
			//Initialize container
			p_container.Init(settings.getAssetLocation("particle_image"),
				settings.getAssetLocation("vertex_shader"),
				settings.getAssetLocation("pixel_shader"));
		
/*
			GLuint colorTex, depthTex, fbo;
			// create a RGBA color texture
			glGenTextures(1, &colorTex);
			glBindTexture(GL_TEXTURE_2D, colorTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
				1024, 768,
				0, GL_RGBA, GL_UNSIGNED_BYTE,
				NULL);

			// create a depth texture
			glGenTextures(1, &depthTex);
			glBindTexture(GL_TEXTURE_2D, depthTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
				1024, 768,
				0, GL_DEPTH_COMPONENT, GL_FLOAT,
				NULL);

			// create the framebuffer object
			glGenFramebuffers(1, &fbo);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

			// attach color
			glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorTex, 0);
			glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTex, 0);
*/
			int FRAME_COUNT = 30; //number of frames in gif
			for (int i = 0; i < FRAME_COUNT; i++)
			{
				// bind the framebuffer as the output framebuffer
			//	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

				c_loader.angle += 0.01;
				p_container.UpdateParticles(0.1); //convert ms to s

				float tmp;
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();
				gluLookAt(0.f, 0.f, 3.f, 0.f, 0.f, -5.f, 0.f, 1.f, 0.f);

				// rotate it around the y axis
				glRotatef(c_loader.angle, 0.f, 1.f, 0.f);
				// scale the whole asset to fit into our view frustum 
				tmp = c_loader.scene_max.x - c_loader.scene_min.x;
				tmp = aisgl_max(c_loader.scene_max.y - c_loader.scene_min.y, tmp);
				tmp = aisgl_max(c_loader.scene_max.z - c_loader.scene_min.z, tmp);
				tmp = 1.f / tmp;
				glScalef(tmp, tmp, tmp);

				// center the model
				glTranslatef(-c_loader.scene_center.x, -c_loader.scene_center.y, -c_loader.scene_center.z);
				GLuint scene_list = glGenLists(1);
				glNewList(scene_list, GL_COMPILE);
				c_loader.render(); //render
				glEndList();
				glCallList(scene_list);

				p_container.Draw();

				// bind the framebuffer as the output framebuffer
				std::string str = make_filename("../../output/images/simple_screencap/",
					"frame","bmp");
				save_screenshot(1024, 768, (char*)str.c_str());
			}
		}
		TEST_METHOD(Test3DModelValid)
		{
			//Load 3D model
			ColladaLoader c_loader;
			GlobalSettings settings;

			Assert::IsTrue(settings.LoadFromJson("../../assets/settings.json"));

			char* model_name = settings.getAssetLocation("model_name");
			c_loader.loadasset(model_name);

			const aiScene* scene = c_loader.getSceneObject();
			Assert::IsTrue(scene != NULL);

			//The 3D model must be a closed, orientable manifold.
			//We test it is closed by checking each edge is adjacent to 
			//two faces. To check it is manifold, we test that every 
			//vertex is surrounded by a fan (closed) of faces. To check
			//it is orientable, we check the orientation of each edge
			//on each face are compatible.

			for (int i = 0; i < scene->mNumMeshes; i++)
			{
				aiMesh* current_mesh = scene->mMeshes[i];

				//Attempt to build adjacancy graph from each mesh
				MeshAdjacancyGraph graph;
				Assert::IsTrue(graph.buildGraph(current_mesh));
				
				//Check each vertex is surrounded by a fan of faces.
				Assert::IsTrue(graph.vertexFanProperty());
			}


		}
		TEST_METHOD(TestParticleGLIntegration)
		{
			GLInitTesting();

			ParticleContainer p_container;
			GlobalSettings settings;

			//Load 3D model
			Assert::IsTrue(settings.LoadFromJson("../../assets/settings.json"));
			//Initialize container
			p_container.Init(settings.getAssetLocation("particle_image"),
				settings.getAssetLocation("vertex_shader"),
				settings.getAssetLocation("pixel_shader"));

			int initialization_time = glutGet(GLUT_ELAPSED_TIME);

			//check initalization takes no more than 3 seconds
			//Assert::IsTrue(initialization_time < 1000);

			int frame_count = 10;
			int particles_per_frame = 0;
			int particles_count = 0;
			for (int i = 0; i < frame_count; i++)
			{
				p_container.UpdateParticles(0.02);
				p_container.Draw();
				particles_per_frame = p_container.getParticleCount() - particles_count;
				particles_count += particles_per_frame;
			}
			int total_frame_time = glutGet(GLUT_ELAPSED_TIME) - initialization_time;

			//check framerate doesnt dip below 60 fps
			Assert::IsTrue(total_frame_time/frame_count < 1000/60);

			//check particles are being created and at a reasonable speed
			Assert::IsTrue(particles_per_frame > 0 );
			Assert::IsTrue(particles_per_frame < 1000);

			//check particles move at a reasonable speed (check for numerical instability)
			Assert::IsTrue(p_container.getAverageSpeed() < 10);

		}

	};
}