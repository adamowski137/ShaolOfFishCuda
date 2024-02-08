#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include "CudaFish.cuh"
#include "cuda_runtime.h"
#include "thrust/random.h"
#include "thrust/device_ptr.h"
#include "thrust/transform.h"
#include "thrust/copy.h"
#include "thrust/gather.h"
#include "thrust/sort.h"
#include <chrono>
#include <device_launch_parameters.h>
#define getDistanceSQ(x1, y1, x2, y2)	(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// zakomentowane zosta³y fragmenty kodu s³u¿¹ce do obliczania czasu
// aby nie zaburzaæ czasu trwania obliczeñ



__device__ __host__ unsigned int positionToSquare(float x, float y, int amountOfSquaresRow, float squareWidth)
{
	// map so that values are in range (0, 2)
	x += 1.0f;
	y -= 1.0f;
	y *= -1;

	// minor adjustments so we dont go out of bounds
	if (x < 0.0f) x = 0.0f;
	if (y < 0.0f) y = 0.0f;
	if (x >= 2.0f) x = 1.999999f;
	if (y >= 2.0f) y = 1.999999f;

	// calculate square coordinates
	unsigned int xIdx = x / squareWidth;
	unsigned int yIdx = y / squareWidth;

	// map it to 1D
	return yIdx * amountOfSquaresRow + xIdx;
}

// function that fills two arrays one with index of the fish second with grid index of the fish
__global__ void computeIndicesKern(unsigned int amountOfFish, int amountOfSquaresRow, float squareWidth,
	float* x, float* y, unsigned int* indices, unsigned int* grid_indices)
{
	const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= amountOfFish) return;

	unsigned int gidx = positionToSquare(x[index], y[index], amountOfSquaresRow, squareWidth);
	grid_indices[index] = positionToSquare(x[index], y[index], amountOfSquaresRow, squareWidth);

	indices[index] = index;
}

// utility function used to fill grid_start and grid_end arrays
__global__ void resetIntBufferKern(unsigned int gridSize, int* intBuffer, int value)
{
	const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= gridSize) return;
	intBuffer[index] = value;
}

// function responsible for finding first and last fish corresponding to each grid cell
__global__ void identifyCellStartEndKern(
	unsigned int amountOfFish, unsigned int* fish_to_grid,
	int* grid_cell_start, int* grid_cell_end)
{
	const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= amountOfFish) return;

	const auto current_grid_index = fish_to_grid[index];

	if (index == 0) {
		grid_cell_start[current_grid_index] = 0;
		return;
	}
	if (index == amountOfFish - 1)
	{
		grid_cell_end[current_grid_index] = amountOfFish - 1;
	}

	const unsigned int prev_grid_index = fish_to_grid[index - 1];
	if (current_grid_index != prev_grid_index) {
		grid_cell_end[prev_grid_index] = static_cast<int>(index - 1);
		grid_cell_start[current_grid_index] = index;

		if (index == amountOfFish - 1) { grid_cell_end[current_grid_index] = index; }
	}
}

// taken from internet hash function
__host__ __device__ unsigned int hash(unsigned int a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

// function that initializes position and velocity arrays with random values
__global__ void generateRandomPositionsKernel(int amountOfFish, float* x, float* y, float* vx, float* vy)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= amountOfFish) return;

	thrust::default_random_engine rng{ hash(static_cast<int>(static_cast<float>(index))) };
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);
	x[index] = unitDistrib(rng);
	y[index] = unitDistrib(rng);
	vx[index] = unitDistrib(rng) / 10.0f;
	vy[index] = unitDistrib(rng) / 10.0f;
}

// function thats responsible for copying data to a array thats mapped to a vbo
__global__ void copyDataToVBO(int amountOfFish, float* x, float* vbo)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amountOfFish) return;
	vbo[index] = x[index];
}
// function responsible for calculating new velocities
__global__ void updateVelocitiesKernel(const unsigned int amountOfFish, int amountOfSquaresRow, float safeZoneRadiusSQ, float viewZoneRadiusSQ,
	float squareWidth, float maxSpeed, float minSpeed, float avoidFactor, float centeringFactor, float matchingfactor, float turnfactor, float margin,
	float xMouse, float  yMouse, int* grid_cell_start, int* grid_cell_end, unsigned int* fish_mapping,
	float* x, float* y, float* vx1, float* vy1, float* vx2, float* vy2)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= amountOfFish) return;

	// declare all needed variables
	float cx = x[index];
	float cy = y[index];
	float cvx = vx1[index];
	float cvy = vy1[index];

	unsigned int numberOfNeighbours = 0;
	unsigned int numberOfSafeZone = 0;

	float xAvgPos = 0.0f;
	float yAvgPos = 0.0f;

	float xAvgVel = 0.0f;
	float yAvgVel = 0.0f;

	float avoidx = 0.0f;
	float avoidy = 0.0f;

	// calculate which cells should be checked
	int gridIndex = positionToSquare(x[index], y[index], amountOfSquaresRow, squareWidth);
	int gx = gridIndex % amountOfSquaresRow;
	int gy = gridIndex / amountOfSquaresRow;
	const auto x_begin = max(gx - 1, 0);
	const auto x_end = min(gx + 1, amountOfSquaresRow - 1);
	const auto y_begin = max(gy - 1, 0);
	const auto y_end = min(gy + 1, amountOfSquaresRow - 1);

	for (int i = x_begin; i <= x_end; i++)
	{
		for (int j = y_begin; j <= y_end; j++)
		{
			// find cells corresponding starting fish and last fish
			int nIdx = i + j * amountOfSquaresRow;
			int first = grid_cell_start[nIdx];
			int last = grid_cell_end[nIdx];
			if (first == -1) continue;
			for (int k = first; k <= last; k++)
			{
				int fish = fish_mapping[k];
				if (fish == index) continue;
				float fx = x[fish];
				float fy = y[fish];
				float fvx = vx1[fish];
				float fvy = vy1[fish];
				// calculate the fish input on final velocity
				float distSQ = getDistanceSQ(x[index], y[index], fx, fy);
				if (distSQ < safeZoneRadiusSQ)
				{
					avoidx += (cx - fx);
					avoidy += (cy - fy);
				}

				if (distSQ < viewZoneRadiusSQ)
				{
					xAvgPos += fx;
					yAvgPos += fy;
					numberOfNeighbours++;

					xAvgVel += fvx;
					yAvgVel += fvy;
				}
			}
		}
	}

	// if there were any neighbours calculate velocity input generated by them
	if (numberOfNeighbours > 0)
	{
		xAvgPos = (xAvgPos / numberOfNeighbours) - cx;
		yAvgPos = (yAvgPos / numberOfNeighbours) - cy;

		xAvgVel = (xAvgVel / numberOfNeighbours) - cvx;
		yAvgVel = (yAvgVel / numberOfNeighbours) - cvy;
	}

	// calculate new velocity
	float vx = cvx + avoidx * avoidFactor + (xAvgPos) * centeringFactor + (xAvgVel) * matchingfactor;
	float vy = cvy + avoidy * avoidFactor + (yAvgPos) * centeringFactor + (yAvgVel) * matchingfactor;

	// make sure you stay in bounds of the screen
	if (cx > 1.0f - margin)
	{
		vx -= turnfactor;
	}
	if (cy > 1.0f - margin)
	{
		vy -= turnfactor;
	}
	if (cx < -1.0f + margin)
	{
		vx += turnfactor;
	}
	if (cy < -1.0f + margin)
	{
		vy += turnfactor;
	}

	// handle mouse input
	if (getDistanceSQ(xMouse, yMouse, cx, cy) < viewZoneRadiusSQ)
	{
		vx += (cx - xMouse);
		vy += (cy - yMouse);
	}

	// make sure the velocity is in bounds
	float v = sqrtf(vx * vx + vy * vy);
	if (v > maxSpeed)
	{
		vx = (vx / v) * maxSpeed;
		vy = (vy / v) * maxSpeed;
	}
	if (v < minSpeed)
	{
		vx = (vx / v) * minSpeed;
		vy = (vy / v) * minSpeed;
	}

	// save the calculated velocity
	vx2[index] = vx;
	vy2[index] = vy;
}

CudaFishSpecies::CudaFishSpecies(const char* cfgFile, const char* outPath): threads{512}, out{outPath}
{
	// load parameters from config file
	loadData(cfgFile);
	// setup shader data
	shaderSetup();


	blocks = ceilf(amountOfFish / (float)threads);
	squareWidth = 2.0f * sqrtf(viewZoneRadiusSQ);
	gridSideCount = 2.0f / squareWidth;
	
	// declare all device arrays
	setupDeviceData();
}

void CudaFishSpecies::setupDeviceData()
{
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_x), amountOfFish * sizeof(float)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_y), amountOfFish * sizeof(float)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_vx), amountOfFish * sizeof(float)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_vy), amountOfFish * sizeof(float)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_new_vx), amountOfFish * sizeof(float)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_new_vy), amountOfFish * sizeof(float)));

	generateRandomPositionsKernel << <blocks, threads >> > (amountOfFish, dev_x, dev_y, dev_vx, dev_vy);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_fish_mapping), amountOfFish * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_fish_to_grid), amountOfFish * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_grid_cell_start),
		gridSideCount * gridSideCount * sizeof(int)));
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_grid_cell_end),
		gridSideCount * gridSideCount * sizeof(int)));

}

void CudaFishSpecies::shaderSetup()
{
	float* tmpVBO = new float[amountOfFish];
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbox);
	glGenBuffers(1, &vboy);
	glGenBuffers(1, &vbovx);
	glGenBuffers(1, &vbovy);
	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbox);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_STREAM_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer((GLuint)0, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vboy);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_STREAM_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer((GLuint)1, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbovx);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_STREAM_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer((GLuint)2, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbovy);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_STREAM_DRAW);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer((GLuint)3, 1, GL_FLOAT, GL_FALSE, 0, 0);

	cudaGLRegisterBufferObject(vbox);
	cudaGLRegisterBufferObject(vboy);
	cudaGLRegisterBufferObject(vbovx);
	cudaGLRegisterBufferObject(vbovy);

	delete[] tmpVBO;
}

void CudaFishSpecies::loadData(const char* path)
{
	std::ifstream input{ path };

	std::string tmp;
	input >> tmp;
	input >> amountOfFish;
	input >> tmp;
	input >> safeZoneRadiusSQ;
	input >> tmp;
	input >> viewZoneRadiusSQ;
	input >> tmp;
	input >> maxSpeed;
	input >> tmp;
	input >> minSpeed;
	input >> tmp;
	input >> avoidFactor;
	input >> tmp;
	input >> centeringFactor;
	input >> tmp;
	input >> matchingFactor;
	input >> tmp;
	input >> turnFactor;
	input >> tmp;
	input >> margin;
	input >> tmp;
	input >> color.r;
	input >> tmp;
	input >> color.g;
	input >> tmp;
	input >> color.b;
	input.close();
}
CudaFishSpecies::~CudaFishSpecies()
{
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_vx);
	cudaFree(dev_vx);
	cudaFree(dev_new_vx);
	cudaFree(dev_new_vy);
	cudaFree(dev_fish_mapping);
	cudaFree(dev_fish_to_grid);
	cudaFree(dev_grid_cell_start);
	cudaFree(dev_grid_cell_end);

	glDeleteBuffers(1, &vbox);
	glDeleteBuffers(1, &vboy);
	glDeleteBuffers(1, &vbovx);
	glDeleteBuffers(1, &vbovy);
	out.close();
}

void CudaFishSpecies::updatePosition(float xMouse, float yMouse)
{
	//auto start = std::chrono::high_resolution_clock().now();
	
	// compute which grid block corresponds to which fish
	computeIndicesKern<<<blocks, threads>>>(amountOfFish, gridSideCount, squareWidth, dev_x, dev_y, dev_fish_mapping, dev_fish_to_grid);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	thrust::device_ptr<unsigned int> thrust_mapping = thrust::device_pointer_cast(dev_fish_mapping);
	thrust::device_ptr<unsigned int> thrust_grid = thrust::device_pointer_cast(dev_fish_to_grid);
	// sort the fishes based on block they belong to dev_particle_array_indices now is a mapping beetwen the starting indexing and indexing in dev_particle_grid
	thrust::sort_by_key(thrust_grid, thrust_grid + amountOfFish, thrust_mapping);


	// calculate how many blocks is needed
	int blocksCell = ceilf((gridSideCount * gridSideCount) / (float)threads);


	// set the start and end grid array to -1
	resetIntBufferKern<< <blocksCell, threads>> > (gridSideCount * gridSideCount, dev_grid_cell_start, -1);
	resetIntBufferKern << <blocksCell, threads>> > (gridSideCount * gridSideCount, dev_grid_cell_end, -1);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// update the start grid and end grid data
	identifyCellStartEndKern << <blocks, threads >> > (amountOfFish, dev_fish_to_grid, dev_grid_cell_start, dev_grid_cell_end);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//auto end = std::chrono::high_resolution_clock().now();
	//auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//out << "grid setup: " << dur << std::endl;


	thrust::device_ptr<float> thrust_x{ dev_x };
	thrust::device_ptr<float> thrust_y{ dev_y };
	thrust::device_ptr<float> thrust_vx{ dev_vx };
	thrust::device_ptr<float> thrust_vy{ dev_vy };

	//start = std::chrono::high_resolution_clock().now();
	
	// calculate new velocities
	updateVelocitiesKernel << <blocks, threads >> > (amountOfFish, gridSideCount, safeZoneRadiusSQ, viewZoneRadiusSQ, squareWidth,
		maxSpeed, minSpeed, avoidFactor, centeringFactor, matchingFactor, turnFactor, margin, xMouse, yMouse,
		dev_grid_cell_start, dev_grid_cell_end, dev_fish_mapping, dev_x, dev_y, dev_vx,
		dev_vy, dev_new_vx, dev_new_vy);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//end = std::chrono::high_resolution_clock().now();
	//dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//out << "calculate velocity: : " << dur << std::endl;

	thrust::device_ptr<float> thrust_new_vx{ dev_new_vx };
	thrust::device_ptr<float> thrust_new_vy{ dev_new_vy };

	//start = std::chrono::high_resolution_clock().now();

	// update position
	thrust::transform(thrust_x, thrust_x + amountOfFish, thrust_new_vx, thrust_x, thrust::plus<float>());
	thrust::transform(thrust_y, thrust_y + amountOfFish, thrust_new_vy, thrust_y, thrust::plus<float>());
	
	//end = std::chrono::high_resolution_clock().now();
	//dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//out << "update position: " << dur << std::endl;

	// swap velocities
	std::swap(dev_vx, dev_new_vx);
	std::swap(dev_vy, dev_new_vy);
}

void CudaFishSpecies::setShaderData(Shader& shader)
{
	shader.setVec3("color", color);
}

void CudaFishSpecies::renderData()
{
	//auto start = std::chrono::high_resolution_clock().now();
	float *x, *y, *vx, *vy;
	cudaGLMapBufferObject((void**)&x, vbox);
	cudaGLMapBufferObject((void**)&y, vboy);
	cudaGLMapBufferObject((void**)&vx, vbovx);
	cudaGLMapBufferObject((void**)&vy, vbovy);

	copyDataToVBO << <blocks, threads >> > (amountOfFish, dev_x, x);
	copyDataToVBO << <blocks, threads >> > (amountOfFish, dev_y, y);
	copyDataToVBO << <blocks, threads >> > (amountOfFish, dev_vx, vx);
	copyDataToVBO << <blocks, threads >> > (amountOfFish, dev_vy, vy);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaGLUnmapBufferObject(vbox);
	cudaGLUnmapBufferObject(vboy);
	cudaGLUnmapBufferObject(vbovx);
	cudaGLUnmapBufferObject(vbovy);


	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, amountOfFish);

	//auto end = std::chrono::high_resolution_clock().now();
	//auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//out << "Pass data to opengl: " << dur << std::endl;
}