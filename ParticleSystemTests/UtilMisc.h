#include <stdio.h>
#include <string>
#include <IL/il.h>

typedef char BYTE;

std::string make_filename(char* containing_folder, char* folder_name,
	char* file_name, int index, char* ext)
{
	int f_ind = 0;
	bool found_filename = false;
	while (!found_filename)
	{
		char filename[2048];
		sprintf(filename, "%s%s%d/%s0.%s", containing_folder,
			folder_name, f_ind, file_name, ext);
		FILE* pFile = fopen(filename, "w");
		if (pFile)
		{
			fclose(pFile);
			f_ind++;
		}
		else
			found_filename = true;
	}
	char filename[2048];
	sprintf(filename, "%s%s%d/%s%d.%s", containing_folder,
		folder_name, f_ind, file_name,index, ext);
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