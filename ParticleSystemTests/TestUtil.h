#include <stdio.h>
#include <string>
#include <IL/il.h>
#include <fstream>
typedef char BYTE;

void GLInitTesting()
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
}

std::string benchmark_result_format(double* indep_var, int var_count, int time)
{
	std::string output;
	for (int i = 0; i < var_count; i++)
	{
		char temp[512];
		sprintf((char*)temp, "%f, ", indep_var[i]);
		output += temp;
	}

	char temp[512];
	sprintf((char*)temp, "%d ms \n", time);
	output += temp;

	return output;
}

void writeFile(std::string filename, std::string file_contents)
{
	std::ofstream file(filename);
	file.write((char*)file_contents.c_str(), file_contents.length());
	file.close();
}

std::string make_filename(char* containing_folder,
	char* file_name, char* ext)
{
	int f_ind = 0;
	bool found_filename = false;
	while (!found_filename)
	{
		char filename[2048];
		sprintf(filename, "%s%s%d.%s", containing_folder, file_name, f_ind, ext);
		FILE* pFile = fopen(filename, "r");
		if (pFile)
		{
			fclose(pFile);
			f_ind++;
		}
		else
			found_filename = true;
	}
	char filename[2048];
	sprintf(filename, "%s%s%d.%s", containing_folder, file_name, f_ind, ext);
	std::string final_filename(filename);
	return final_filename;
}

void save_screenshot(int width, int height, wchar_t* filename)
{
	ILuint img_size = sizeof(BYTE) * width * height * 3;
	BYTE *raw_img = (BYTE*)malloc(img_size);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, raw_img);

	if (raw_img != NULL)
	{
		//Then save it using devil
		ILuint ImgId = 0;
		ilInit();
		ilGenImages(1, &ImgId);
		ilBindImage(ImgId);

		//I don't know if ilCopyPixel is the right function to call
		ilCopyPixels(0, 0, 0, width, height, 1, IL_RGB, IL_BYTE, raw_img);

		ILboolean error_code = ilSaveImage(filename);
		Assert::IsTrue(error_code = IL_TRUE);

		ilDeleteImages(1, &ImgId);
	}

	free(raw_img);
}