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
	TEST_CLASS(ParticleSystemAccuracy)
	{
	public:
		TEST_METHOD(ParticleSystemSanityChecks)
		{
			ParticleContainer p_container;
			GlobalSettings settings;
			//Load 3D model
			Assert::IsTrue(settings.LoadFromJson("../../assets/settings.json"));
			p_container.UpdateParticles(10);

			//check running process
			int test_frame_count = 100;
			for (int i = 0; i < test_frame_count; i++)
			{
				std::vector<Particle> p_list = p_container.getAll();
				p_container.UpdateParticles(0.2);

				for (int j = 0; j < p_container.max_particle_count; j++)
				{
					//check acceleration
				//	p_list[j].lambda
				}
			};

		}
		TEST_METHOD(GeneratePosReport)
		{
			ParticleContainer p_container;
			GlobalSettings settings;
			//Load 3D model
			Assert::IsTrue(settings.LoadFromJson("../../assets/settings.json"));
			ColladaLoader model;
			model.loadasset(settings.getAssetLocation("model_name"));
			p_container.SetObstacle(&model);

			int test_frame_count = 100;
			for (int i = 0; i < test_frame_count; i++)
			{
				std::string output = p_container.livePositionsList();
				std::string filename = make_filename("../../output/reports/particle_positions/",
					"simple_timer", "csv"); 
				writeFile(filename, output);
				p_container.UpdateParticles(0.02);
			}
		}
	};
}