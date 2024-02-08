#pragma once
#include "Shader.hpp"

class FishBase 
{
public:
	void virtual updatePosition(float xMouse, float yMouse) = 0;
	void virtual setShaderData(Shader& shader) = 0;
	void virtual renderData() = 0;

	float safeZoneRadiusSQ;
	float viewZoneRadiusSQ;
	float maxSpeed;
	float minSpeed;
	float avoidFactor;
	float centeringFactor;
	float matchingFactor;
	float turnFactor;
	float margin;
};