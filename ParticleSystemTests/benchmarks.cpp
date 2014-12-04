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
	TEST_CLASS(Benchmarks)
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
	};
}