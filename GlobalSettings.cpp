#include "GlobalSettings.h"
#include <fstream>

bool GlobalSettings::LoadFromJson(char* settings)
{
	std::ifstream settings_stream(settings, std::ifstream::binary);

	bool parsingSuccessful = reader.parse(settings_stream, root, false);
	if (!parsingSuccessful)
	{
		// report to the user the failure and their locations in the document.
		printf((char*)reader.getFormattedErrorMessages().c_str());
		return false;
	}
	return true;
}
bool GlobalSettings::record_video()
{
	return root["output"]["record"].asBool();
}
char* GlobalSettings::video_file_name(){
	char* s = new char[4096];
	char* string =(char*)root["output"]["jpeg_folder"].asCString();
	memcpy(s, string, 256);

	return s;
}
int GlobalSettings::particle_count(){
	return root["settings"]["particle_count"].asInt();
}
float GlobalSettings::getConstant(char* name)
{
	char* s = new char[256];
	return root["constants"][name].asFloat();
}
char* GlobalSettings::getAssetLocation(char* name)
{
	char* s = new char[256]; 
	char* string = (char*)root["assets"][name].asCString();

	if (string)
		memcpy(s, string, 256);
	else
		s = NULL;

	return s;
}
GlobalSettings::GlobalSettings()
{
}
GlobalSettings::~GlobalSettings()
{
}
