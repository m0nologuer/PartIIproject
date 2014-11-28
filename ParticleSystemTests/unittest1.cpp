#include "stdafx.h"
#include "CppUnitTest.h"
#include <gl/glew.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include "ParticleContainer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ParticleSystemTests
{		
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(TestParticleGLIntegration)
		{

			// initalize glut
			glutInitWindowSize(900, 600);
			glutInitWindowPosition(100, 100);
			glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
			int argc = 0; char* argv = " ";
			glutInit(&argc, &argv);

			glutCreateWindow("Test");

			//initialize glew
			if (glewInit() != 0)
				return;

			glutGet(GLUT_ELAPSED_TIME);
			ParticleContainer p_container;
			p_container.Init("particle.DDS", "Particle.vertexshader", "Particle.fragmentshader");
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