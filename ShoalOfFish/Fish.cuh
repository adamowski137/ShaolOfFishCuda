#include <glm.hpp>
#include <set>
#include "Shader.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FishBase.cuh"
class FishSpecies : public FishBase
{
private:
	float safeZoneRadiusSQ;
	float viewZoneRadiusSQ;
	float viewZoneRadius;
	float maxSpeed;
	float minSpeed;
	float avoidFactor;
	float centeringFactor;
	float matchingfactor;
	float turnfactor;
	float margin;
	glm::vec3 color;
	unsigned int amountOfFish;
	float* fishPositionX;
	float* fishPositionY;
	float* fishVelocityX;
	float* fishVelocityY;
	float* fishNewVelocityX;
	float* fishNewVelocityY;
	std::set<int>* squares;
	unsigned int amountOfSquaresRow;
	int positionToSquare(float x, float y);
	GLuint vao, vbox, vboy, vbovx, vbovy;


	void loadData(const char* path);
	void shaderSetup();
	void setupDeviceData();

	void bufferData();
	void updateFishVelocity(float xMouse, float yMouse);

public:
	FishSpecies(const char* path);
	~FishSpecies();
	void setShaderData(Shader shader) override;
	void renderData() override;
	void updatePosition(float xMouse, float yMouse) override;
};


