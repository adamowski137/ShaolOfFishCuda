#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <gtc/matrix_transform.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include "Window.hpp"
#include "Shader.hpp"
#include "Fish.cuh"
#include "CudaFish.cuh"
GLenum glCheckError_(const char* file, int line)
{
	GLenum errorCode;
	while ((errorCode = glGetError()) != GL_NO_ERROR)
	{
		std::string error;
		switch (errorCode)
		{
		case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
		case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
		case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
		case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
		case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
		case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
		}
		std::cout << error << " | " << file << " (" << line << ")" << std::endl;
	}
	return errorCode;
}

#define glCheckError() glCheckError_(__FILE__, __LINE__) 
float Window::xMouse = -2.0f;
float Window::yMouse = -2.0f;
int Window::WIDTH = 0;
int Window::HEIGHT = 0;
Shader* Window::shader = nullptr;
GLFWwindow* Window::window = nullptr;
bool Window::gpuMode = false;


Window::Window()
{
	char input;
	std::cout << "Gpu/Cpu: (g/c): ";
	std::cin >> input;
	gpuMode = input == 'g';
	if (!glfwInit())
		throw "glfw init error";

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

	WIDTH = mode->width;
	HEIGHT = mode->height;
	window = glfwCreateWindow(mode->width, mode->height, "Shoal of Fish", monitor, NULL);
	
	if (!window)
		throw "glfw create window error";

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, Window::framebuffer_size_callback);
	glfwSetCursorPosCallback(window, Window::cursor_position_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw "Failed to initialize GLAD";
	}


	shader = new Shader{ "vertex.glsl", "geometry.glsl", "fragment.glsl" };
	setOneTimeShaderData();
}

void Window::setOneTimeShaderData()
{
	shader->use();
	shader->setVec3("color", glm::vec3{ 1.0f, 1.0f, 1.0f });
}

void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	WIDTH = width;
	HEIGHT = height;
	glViewport(0, 0, width, height);
}

void Window::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	xMouse = (2 * (xpos / WIDTH)) - 1.0f; 
	yMouse = (2 *  (- ypos / HEIGHT)) + 1.0f;
}


Window::~Window()
{
	delete shader;
	glfwTerminate();
}

void Window::runWindow()
{
	std::vector<FishBase*> fishSpecies{};

	if (gpuMode)
	{
		fishSpecies.push_back(new CudaFishSpecies{ "CudaFish.txt" });
		fishSpecies.push_back(new CudaFishSpecies{ "CudaFish2.txt" });
	}
	else
	{
		fishSpecies.push_back(new FishSpecies{ "Fish.txt" });
		fishSpecies.push_back(new FishSpecies{ "Fish2.txt" });
	}

	shader->use();
	long long prevDur = 0;
	
	while (!glfwWindowShouldClose(window))
	{ 
		glCheckError();
		glClear(GL_COLOR_BUFFER_BIT);
		auto start = std::chrono::high_resolution_clock::now();

		for (auto& fish : fishSpecies)
		{
			fish->setShaderData(*shader);
			fish->updatePosition(xMouse, yMouse);
			fish->renderData();
		}

		glfwSwapBuffers(window);

		glfwPollEvents();

		auto end = std::chrono::high_resolution_clock::now();
		prevDur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		//std::cout << "FPS: " << 1000000 / prevDur << std::endl;

		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
	}
	for (auto& fish : fishSpecies)
	{
		delete fish;
	}
}

Window Window::getInstance()
{
	return Window();
}
