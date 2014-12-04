#include <chrono>
#include <string>
#include <IL/il.h>
#include <fstream>

typedef char BYTE;

void GLInitTesting();
std::string benchmark_result_format(double* indep_var, int var_count, int time);
void writeFile(std::string filename, std::string file_contents);
std::string make_filename(char* containing_folder,
	char* file_name, char* ext);
void save_screenshot(int width, int height, char* filename);