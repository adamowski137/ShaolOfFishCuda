#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Fish.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <chrono>

#define getDistanceSQ(x1, y1, x2, y2)	(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

// zakomentowane zosta³y fragmenty kodu s³u¿¹ce do obliczania czasu
// aby nie zaburzaæ czasu trwania obliczeñ

FishSpecies::FishSpecies(const char* path, const char* outPath): out{outPath}
{
	// load all the parameters from a configuration file
	loadData(path);
	// set up the shader data for drawing
	shaderSetup();

	// allocate memory for variables
	amountOfSquaresRow = 2 * static_cast<unsigned int>(ceilf(2.0f / sqrtf(viewZoneRadiusSQ)));
	fishPositionX = new float[amountOfFish];
	fishPositionY = new float[amountOfFish];
	fishVelocityX = new float[amountOfFish];
	fishVelocityY = new float[amountOfFish];
	fishNewVelocityX = new float[amountOfFish];
	fishNewVelocityY = new float[amountOfFish];
	// this is needed for cpu version of grid
	// each square will have a seperate set of fishes that are in this square. 
	squares = new std::set<int>[amountOfSquaresRow * amountOfSquaresRow];

	// generate random positions and velocities for fishes
	for (unsigned int i = 0; i < amountOfFish; i++)
	{
		float x = ((rand() % 200) - 100) / 100.0f;
		float y = ((rand() % 200) - 100) / 100.0f;
		float vx = ((rand() % 200) - 100) / 1000.0f;
		float vy = ((rand() % 200) - 100) / 1000.0f;
		int idx = positionToSquare(x, y);

		fishPositionX[i] = x;
		fishPositionY[i] = y;
		fishVelocityX[i] = vx;
		fishVelocityY[i] = vy;

		squares[idx].insert(i);
	}
}

FishSpecies::~FishSpecies()
{
	out.close();
	delete[] fishPositionX;
	delete[] fishPositionY;
	delete[] fishVelocityX;
	delete[] fishVelocityY;
	delete[] fishNewVelocityX;
	delete[] fishNewVelocityY;
	delete[] squares;
}

void FishSpecies::shaderSetup()
{
	float* tmpVBO = new float[amountOfFish];
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbox);
	glGenBuffers(1, &vboy);
	glGenBuffers(1, &vbovx);
	glGenBuffers(1, &vbovy);
	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbox);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer((GLuint)0, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vboy);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer((GLuint)1, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbovx);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer((GLuint)2, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbovy);
	glBufferData(GL_ARRAY_BUFFER, (amountOfFish) * sizeof(GLfloat), tmpVBO, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer((GLuint)3, 1, GL_FLOAT, GL_FALSE, 0, 0);

	delete[] tmpVBO;
}

void FishSpecies::loadData(const char* path)
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
	input >> matchingfactor;
	input >> tmp;
	input >> turnfactor;
	input >> tmp;
	input >> margin;
	input >> tmp;
	input >> color.r;
	input >> tmp;
	input >> color.g;
	input >> tmp;
	input >> color.b;
	input.close();
	viewZoneRadius = sqrtf(viewZoneRadiusSQ);
}

void FishSpecies::renderData()
{
	bufferData();
	glDrawArrays(GL_POINTS, 0, amountOfFish);
}

void FishSpecies::setShaderData(Shader& shader)
{
	shader.setVec3("color", color);
}

void FishSpecies::updateFishVelocity(float xMouse, float yMouse)
{
	// iterate over every fish
	for (unsigned int i = 0; i < amountOfFish; i++)
	{
		// declare variables
		float x1 = fishPositionX[i];
		float y1 = fishPositionY[i];

		float vx1 = fishVelocityX[i];
		float vy1 = fishVelocityY[i];

		float xAvgPos = 0.0f;
		float yAvgPos = 0.0f;

		float xAvgVel = 0.0f;
		float yAvgVel = 0.0f;

		float avoidx = 0.0f;
		float avoidy = 0.0f;

		unsigned int numberOfNeighbours = 0;

		// calculate which square it belongs to
		int idx = positionToSquare(x1, y1);
		int xIdx = idx % amountOfSquaresRow;
		int yIdx = idx / amountOfSquaresRow;

		// iterate over neighbouring squares
		const int x_begin = xIdx - 1 > 0 ? xIdx - 1 : 0;
		const int x_end = (xIdx + 1) < amountOfSquaresRow - 1 ? xIdx + 1 : amountOfSquaresRow - 1;
		const int y_begin = yIdx - 1 > 0 ? yIdx - 1 : 0;
		const int y_end = (yIdx + 1) < amountOfSquaresRow - 1 ? yIdx + 1 : amountOfSquaresRow - 1;

		for (int xi = x_begin; xi <= x_end; xi++)
		{
			for (int yj = y_begin; yj <= y_end; yj++)
			{
				int sq = xi + yj * amountOfSquaresRow;
				// iterate over every fish in this square
				for (auto fish : squares[sq])
				{
					// make sure you are not counting the same fish
					if (fish == i) continue;

					float x2 = fishPositionX[fish];
					float y2 = fishPositionY[fish];

					float vx2 = fishVelocityX[fish];
					float vy2 = fishVelocityY[fish];

					// check if fish is in the safe zone
					float distSQ = getDistanceSQ(x1, y1, x2, y2);
					if (distSQ < safeZoneRadiusSQ)
					{
						avoidx += (x1 - x2);
						avoidy += (y1 - y2);
					}

					// check if fish is in the view zone
					if (distSQ < viewZoneRadiusSQ)
					{
						xAvgPos += x2;
						yAvgPos += y2;
						numberOfNeighbours++;

						xAvgVel += vx2;
						yAvgVel += vy2;
					}
				}
			}
		}

		// if there are any fishes nearby calculate their average position and velocity
		if (numberOfNeighbours > 0)
		{
			xAvgPos = (xAvgPos / numberOfNeighbours) - x1;
			yAvgPos = (yAvgPos / numberOfNeighbours) - y1;

			xAvgVel = (xAvgVel / numberOfNeighbours) - vx1;
			yAvgVel = (yAvgVel / numberOfNeighbours) - vy1;
		}

		// calculate new velocities
		float vx = fishVelocityX[i] + avoidx * avoidFactor + (xAvgPos) * centeringFactor + (xAvgVel) * matchingfactor;
		float vy = fishVelocityY[i] + avoidy * avoidFactor + (yAvgPos) * centeringFactor + (yAvgVel) * matchingfactor;

		// adjust so that fish dont go out of bounds
		if (x1 > 1.0f - margin)
		{
			vx -= turnfactor;
		}
		if (y1 > 1.0f - margin)
		{
			vy -= turnfactor;
		}
		if (x1 < -1.0f + margin)
		{
			vx += turnfactor;
		}
		if (y1 < -1.0f + margin)
		{
			vy += turnfactor;
		}

		// add escaping from mouse curosor
		if (getDistanceSQ(xMouse, yMouse, x1, y1) < viewZoneRadiusSQ)
		{
			vx += (x1 - xMouse);
			vy += (y1 - yMouse);
		}

		// adjust so that the speed is in bounds
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

		// save our new calculated velocity
		fishNewVelocityX[i] = vx;
		fishNewVelocityY[i] = vy;
	}
	// set newly calculated velocities
	std::swap(fishVelocityX, fishNewVelocityX);
	std::swap(fishVelocityY, fishNewVelocityY);
}

int FishSpecies::positionToSquare(float x, float y)
{
	x += 1.0f;
	y -= 1.0f;
	y *= -1;

	if (x < 0.0f) x = 0.0f;
	if (y < 0.0f) y = 0.0f;
	if (x >= 2.0f) x = 1.99999f;
	if (y >= 2.0f) y = 1.99999f;

	float unit = 2 * sqrtf(viewZoneRadiusSQ);
	
	int xIdx = x / unit;
	int yIdx = y / unit;

	return yIdx * amountOfSquaresRow + xIdx;
}

void FishSpecies::bufferData()
{
	// function responsible for passing data to shaders

	glBindBuffer(GL_ARRAY_BUFFER, vbox);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * amountOfFish, fishPositionX, GL_STREAM_DRAW);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, vboy);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * amountOfFish, fishPositionY, GL_STREAM_DRAW);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, vbovx);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * amountOfFish, fishVelocityX, GL_STREAM_DRAW);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, vbovy);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * amountOfFish, fishVelocityY, GL_STREAM_DRAW);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(3);

	glBindVertexArray(vao);
}

void FishSpecies::updatePosition(float xMouse, float yMouse)
{
	//auto start = std::chrono::high_resolution_clock().now();

	// calculate fish new velocity
	updateFishVelocity(xMouse, yMouse);

	//auto end = std::chrono::high_resolution_clock().now();
	//auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//out << "Calculate velocity: " << dur << std::endl;

	//start = std::chrono::high_resolution_clock().now();
	
	// and update them accordingly
	for (unsigned int i = 0; i < amountOfFish; i++)
	{
		int prevIdx = positionToSquare(fishPositionX[i], fishPositionY[i]);
		fishPositionX[i] += fishVelocityX[i];
		fishPositionY[i] += fishVelocityY[i];
		int newIdx = positionToSquare(fishPositionX[i], fishPositionY[i]);

		if (prevIdx != newIdx)
		{
			squares[prevIdx].erase(squares[prevIdx].find(i));
			squares[newIdx].insert(i);
		}
	}
	//end = std::chrono::high_resolution_clock().now();
	//dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	//out << "Update postion: " << dur << std::endl;
}
