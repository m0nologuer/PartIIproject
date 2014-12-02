#pragma once
#include <string>
#include "json/json.h"

class GlobalSettings
{
	Json::Value root;
	Json::Reader reader;
public:
	//initialize
	bool LoadFromJson(char* settings);

	// asset getter
	char* getAssetLocation(char* name);

	GlobalSettings();
	~GlobalSettings();
};

