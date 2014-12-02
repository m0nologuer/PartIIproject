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
char* GlobalSettings::getAssetLocation(char* name)
{
	char* s = new char[2048];
	char* string = (char*)root["assets"][name].asCString();

	memcpy(s, string, 2048);

	return s;
}
GlobalSettings::GlobalSettings()
{
}
GlobalSettings::~GlobalSettings()
{
}
