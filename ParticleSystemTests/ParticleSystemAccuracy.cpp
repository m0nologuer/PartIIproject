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
	TEST_CLASS(ParticleSystemAccuracy)
	{
	public:
		TEST_METHOD(GeneratePosReport)
		{
			ParticleContainer p_container;
			GlobalSettings settings;
			//Load 3D model
			Assert::IsTrue(settings.LoadFromJson("../../assets/settings.json"));

			int test_frame_count = 100;
			for (int i = 0; i < test_frame_count; i++)
			{
				p_container.UpdateParticles(0.2);
				std::string output = p_container.livePositionsList();
				std::string filename = make_filename("../../output/reports/particle_positions/",
					"simple_timer", "csv");
				writeFile(filename, output);
			}
		}
	};
}