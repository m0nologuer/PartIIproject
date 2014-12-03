#include "stdafx.h"
#include <gl/glew.h>
#include "MeshAdjacancyGraph.h"
#include "COLLADALoader.h"
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
		TEST_METHOD(BenchmarkKDTree)
		{
			//benchmark boilerplate
			double argv[2];
			int argc = 2;
			int table_size = 50;
			std::string written_output = "particles, radius, timer \n";

			for (int run = 0; run < table_size; run++)
			{
				//set up variables for this iteration
				argv[0] = 500 * (run % 10 + 1);
				argv[1] = (run / 10 + 1) * 10;

				int dummy_count = (int)argv[0];
				double r = argv[1];

				//gnerate dummy particles
				Particle* dummy_particles = new Particle[dummy_count];
				for (int i = 0; i < dummy_count; i++)
				{
					dummy_particles[i].pos = Particle::vec3(rand() % 100,
						rand() % 100, rand() % 100);
					dummy_particles[i].life = 1;
				}
				//start timer
				auto start_time = std::chrono::high_resolution_clock::now();

				//build tree
				KDTree tree(dummy_particles, dummy_count);
				//find all particles with a distance of 20
				tree.findNeighbouringParticles(dummy_particles[0], 50);

				//record time elapsed
				auto end_time = std::chrono::high_resolution_clock::now();
				auto elapsed = end_time - start_time;
				int time = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

				written_output += benchmark_result_format(argv, argc, time);
			}

			//write output
			std::string filename = make_filename("../../output/reports/KD_benchmark/", "simple_timer",
				 "csv");
			writeFile(filename, written_output);

		}
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
		
			int FRAME_COUNT = 30; //number of frames in gif
			for (int i = 0; i < FRAME_COUNT; i++)
			{
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

				std::string str = make_filename("../../output/images/simple_screencap",
					"frame","png");
				std::wstring widestr = std::wstring(str.begin(), str.end());
				save_screenshot(1024, 768, (wchar_t*)widestr.c_str());
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
		TEST_METHOD(TestParticleSystemAccuracy)
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

			int test_frame_count = 100;
			for (int i = 0; i < test_frame_count; i++)
			{
				p_container.UpdateParticles(0.02);
				std::string output;// = p_container.livePositionsList;
				std::string filename = make_filename("../../output/reports/particle_positions",
					"simple_timer","txt");
				writeFile(filename, output);
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