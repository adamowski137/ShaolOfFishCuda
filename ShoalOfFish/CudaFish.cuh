#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include <glm.hpp>
#include "Shader.hpp"
#include "FishBase.cuh"
#include <fstream>
class CudaFishSpecies : public FishBase
{
private:
	float* dev_x;
	float* dev_y;
	float* dev_vx;
	float* dev_vy;

	float* dev_new_vx;
	float* dev_new_vy;

	unsigned int* dev_fish_mapping;
	unsigned int* dev_fish_to_grid;

	int* dev_grid_cell_start;
	int* dev_grid_cell_end;

	glm::vec3 color;
	unsigned int amountOfFish;

	float squareWidth;
	unsigned int gridSideCount;

	const int threads;
	int blocks;
	std::ofstream out;

	GLuint vao, vbox, vboy, vbovx, vbovy;

	void loadData(const char* path);
	void shaderSetup();
	void setupDeviceData();

public:
	CudaFishSpecies(const char* cfgFile, const char* outPath);
	~CudaFishSpecies();
	void updatePosition(float xMouse, float yMouse) override;
	void renderData() override;
	void setShaderData(Shader& shader) override;
};

__device__ __host__ unsigned int positionToSquare(float x, float y, int amountOfSquaresRow, float viewZoneRadius);
