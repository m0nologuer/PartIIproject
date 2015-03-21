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
#include <fstream>
#include <gl/glew.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include "ParticleContainer.h"
#include "GlobalSettings.h"

ColladaLoader c_loader;
ParticleContainer p_container;
GlobalSettings settings;

GLuint scene_list;

CvVideoWriter *writer = NULL;
int w = 900; int h = 600;

// ----------------------------------------------------------------------------
void do_motion(void)
{
	static GLint prev_time = 0;
	static GLint prev_fps_time = 0;
	static int frames = 0;

	int time = glutGet(GLUT_ELAPSED_TIME);
	c_loader.angle += (time - prev_time)*0.1;
	printf("%d cycle time\n", time - prev_time);
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
// 	glTranslatef(0., 0.1f, 0.f);

	// rotate it around the y axis
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	aiMatrix4x4 inverse_mat(matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7], matrix[8],
		matrix[9], matrix[10], matrix[11], matrix[12], matrix[13], matrix[14], matrix[15]);
	matrix[0] = inverse_mat.a1; matrix[1] = inverse_mat.a2; matrix[2] = inverse_mat.a3; matrix[3] = inverse_mat.a4;
	matrix[4] = inverse_mat.b1; matrix[5] = inverse_mat.b2; matrix[6] = inverse_mat.b3; matrix[7] = inverse_mat.b4;
	matrix[8] = inverse_mat.c1; matrix[9] = inverse_mat.c2; matrix[10] = inverse_mat.c3; matrix[11] = inverse_mat.c4;
	matrix[12] = inverse_mat.d1; matrix[13] = inverse_mat.d2; matrix[14] = inverse_mat.d3; matrix[15] = inverse_mat.d4;
	p_container.setMatrix(matrix);
	
	// if the display list has not been made yet, create a new one and
	// fill it with scene contents
	if (scene_list == 0) {
		scene_list = glGenLists(1);
		glNewList(scene_list, GL_COMPILE);
		c_loader.render(); //render
		glEndList();
	}

//	glCallList(scene_list);

	p_container.DrawPoints();


	//for video
	if (writer)
	{
		unsigned char *raw_image = (unsigned char*)calloc(w * h * 3, sizeof(char));
		glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, raw_image);
		IplImage* img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
		img->imageData = (char *)raw_image;
		cvWriteFrame(writer, img);      // add the frame to the file
		cvReleaseImage(&img);
	}
	glutSwapBuffers();

	do_motion();
}

// ----------------------------------------------------------------------------
void reshape(int width, int height)
{

	const double aspectRatio = (float)width / height, fieldOfView = 45.0;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glScalef(1, aspectRatio, 1);
	/*
	gluPerspective(fieldOfView, aspectRatio,
		0.0001, 1000.0);  /* Znear and Zfar  
	glViewport(0, 0, width, height); 
	*/
}
 
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	struct aiLogStream stream;

 	glutInitWindowSize(w,h);
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

	if (settings.record_video())
	{
		// creates video: file to write -- codec that's gonna be used -- frame per second -- size of video frames -- grayscale or not
		CvSize size = cvSize(w, h);
		writer = cvCreateVideoWriter(settings.video_file_name(), CV_FOURCC('I', 'Y', 'U', 'V'), 0, size, true);
	}

	// the model name can be specified in the settings.
	char* model_name = settings.getAssetLocation("model_name");
	c_loader.loadasset(model_name);
	p_container.Set(&c_loader, NULL);
	p_container.setConstants(settings.getConstant("viscosity"), settings.getConstant("buoyancy"));
		 
	glClearColor(0.02f,0.02f,0.05f,1.f);

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
		settings.getAssetLocation("pixel_shader"), settings.getAssetLocation("force_image"));

	glutGet(GLUT_ELAPSED_TIME);
	glutMainLoop();


	cvReleaseVideoWriter(&writer);

	return 0;
}
