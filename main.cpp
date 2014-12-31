// ----------------------------------------------------------------------------
// Simple sample to prove that Assimp is easy to use with OpenGL.
// It takes a file name as command line parameter, loads it using standard
// settings and displays it.
//
// If you intend to _use_ this code sample in your app, do yourself a favour 
// and replace immediate mode calls with VBOs ...
//
// The vc8 solution links against assimp-release-dll_win32 - be sure to
// have this configuration built.
// ----------------------------------------------------------------------------

#include <stdlib.h>
#include <stdio.h>
#include <gl/glew.h>
#include <GL/glut.h>
#include <GL/glext.h>

#include "ParticleContainer.h"
#include "GlobalSettings.h"

ColladaLoader c_loader;
ParticleContainer p_container;
GlobalSettings settings;

GLuint scene_list;

// ----------------------------------------------------------------------------
void do_motion(void)
{
	static GLint prev_time = 0;
	static GLint prev_fps_time = 0;
	static int frames = 0;

	int time = glutGet(GLUT_ELAPSED_TIME);
	c_loader.angle += (time - prev_time)*0.01;
	p_container.UpdateParticles(0.02); //fixed time interval
	prev_time = time;

	frames += 1;
	if ((time - prev_fps_time) > 1000) // update every seconds
	{
		int current_fps = frames * 1000 / (time - prev_fps_time);
		printf("%d fps\n", current_fps);
		frames = 0;
		prev_fps_time = time;
	}


	glutPostRedisplay();
}
// ----------------------------------------------------------------------------
void display(void)
{
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

	// if the display list has not been made yet, create a new one and
	// fill it with scene contents
	if (scene_list == 0) {
		scene_list = glGenLists(1);
		glNewList(scene_list, GL_COMPILE);
		c_loader.render(); //render
		glEndList();
	}

	//glCallList(scene_list);

	p_container.Draw();
	
	glutSwapBuffers();

	do_motion();
}

// ----------------------------------------------------------------------------
void reshape(int width, int height)
{
	const double aspectRatio = (float)width / height, fieldOfView = 45.0;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, aspectRatio,
		1.0, 1000.0);  /* Znear and Zfar */
	glViewport(0, 0, width, height);
}

// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	struct aiLogStream stream;

	glutInitWindowSize(900,600);
	glutInitWindowPosition(100,100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInit(&argc, argv);

	glutCreateWindow("PartII Particle System");
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	//load settings
	bool settings_loaded;
	if (argc >= 2)
		settings_loaded = settings.LoadFromJson(argv[1]);
	else
		settings_loaded = settings.LoadFromJson("../../assets/settings.json");
	if (!settings_loaded) { return 1; };

	// the model name can be specified in the settings.
	char* model_name = settings.getAssetLocation("model_name");
	c_loader.loadasset(model_name);
	//p_container.SetObstacle(&c_loader);

	glClearColor(0.1f,0.1f,0.1f,1.f);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);    // Uses default lighting parameters

	glEnable(GL_DEPTH_TEST);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);

	// XXX docs say all polygons are emitted CCW, but tests show that some aren't.
	if(getenv("MODEL_IS_BROKEN"))  
		glFrontFace(GL_CW);

	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	//initialize glew
	if (glewInit() != 0)
		return 1;

	p_container.Init(settings.getAssetLocation("particle_image"),
		settings.getAssetLocation("vertex_shader"),
		settings.getAssetLocation("pixel_shader"));

	glutGet(GLUT_ELAPSED_TIME);
	glutMainLoop();

	return 0;
}
