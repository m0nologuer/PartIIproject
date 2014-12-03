#include <gl/glew.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include "CppUnitTest.h"
#include <chrono>
#include <stdio.h>
#include <string>
#include <IL/il.h>
#include <fstream>

typedef char BYTE;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

void GLInitTesting();
std::string benchmark_result_format(double* indep_var, int var_count, int time);
void writeFile(std::string filename, std::string file_contents);
std::string make_filename(char* containing_folder,
	char* file_name, char* ext);
void save_screenshot(int width, int height, char* filename);