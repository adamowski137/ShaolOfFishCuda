#include <glad/glad.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
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
int Window::WIDTH = 800;
int Window::HEIGHT = 600;
GLFWwindow* Window::window = nullptr;
bool Window::gpuMode = false;
bool Window::mouseClicked = false;
int Window::selectedFish = 0;

Window::Window()
{
	char input;
	std::cout << "Gpu/Cpu: (g/c): ";
	std::cin >> input;
	gpuMode = input == 'g';
	if (!glfwInit())
		throw "glfw init error";

	window = glfwCreateWindow(WIDTH, HEIGHT, "Shoal of Fish", NULL, NULL);
	
	if (!window)
		throw "glfw create window error";

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);


	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, Window::framebuffer_size_callback);
	glfwSetCursorPosCallback(window, Window::cursor_position_callback);
	glfwSetMouseButtonCallback(window, Window::mouse_button_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw "Failed to initialize GLAD";
	}
	gladLoadGL();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	static ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 450");

}

void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	WIDTH = width;
	HEIGHT = height;
	glViewport(0, 0, width, height);
}

void Window::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (!mouseClicked)	return;
	xMouse = (2 * (xpos / WIDTH)) - 1.0f; 
	yMouse = (2 *  (- ypos / HEIGHT)) + 1.0f;
}
void Window::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		xMouse = (2 * (xpos / WIDTH)) - 1.0f;
		yMouse = (2 * (-ypos / HEIGHT)) + 1.0f;
		mouseClicked = true;
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		xMouse = 10.0f;
		yMouse = 10.0f;
		mouseClicked = false;
	}
}


Window::~Window()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
}

void Window::runWindow()
{
	std::vector<FishBase*> fishSpecies{};

	if (gpuMode)
	{
		fishSpecies.push_back(new CudaFishSpecies{ "CudaFish.txt", "CudaFishOut.txt"});
		fishSpecies.push_back(new CudaFishSpecies{ "CudaFish2.txt", "CudaFishOut2.txt" });
	}
	else
	{
		fishSpecies.push_back(new FishSpecies{ "Fish.txt", "FishOut.txt"});
		fishSpecies.push_back(new FishSpecies{ "Fish2.txt", "FishOut2.txt"});
	}
	Shader shader{ "vertex.glsl", "geometry.glsl", "fragment.glsl" };
	shader.use();
	long long longDur = 1;
	long long shortDur = 1;

	while (!glfwWindowShouldClose(window))
	{ 
		auto start = std::chrono::high_resolution_clock::now();
		glfwPollEvents();

		float avoidParameter = fishSpecies[selectedFish]->avoidFactor;
		float centeringParameter = fishSpecies[selectedFish]->centeringFactor;
		float matchingParameter = fishSpecies[selectedFish]->matchingFactor;
		ImGui_ImplGlfw_NewFrame();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin("Parameters");
		ImGui::SliderInt("Select Fish Species", &selectedFish, 0, fishSpecies.size() - 1);
		ImGui::SliderFloat("Avoid parametr", &avoidParameter, 0.001f, 1.0f);
		ImGui::SliderFloat("Centering parametr", &centeringParameter, 0.001f, 0.01f);
		ImGui::SliderFloat("Matching parametr", &matchingParameter, 0.001f, 1.0f);
		ImGui::Text("Displaying Fps(monitor frequency cap): %d Algortihm Fps: %d", 1000000/longDur, 1000000 / shortDur);
		ImGui::End();
		ImGui::EndFrame();
		ImGui::Render();

		fishSpecies[selectedFish]->avoidFactor = avoidParameter;
		fishSpecies[selectedFish]->centeringFactor = centeringParameter;
		fishSpecies[selectedFish]->matchingFactor = matchingParameter;
		
		glCheckError();
		glClear(GL_COLOR_BUFFER_BIT);
		for (auto& fish : fishSpecies)
		{
			shader.use();
			fish->setShaderData(shader);
			fish->updatePosition(xMouse, yMouse);
			fish->renderData();
		}
		auto shortend = std::chrono::high_resolution_clock::now();
		shortDur = std::chrono::duration_cast<std::chrono::microseconds>(shortend - start).count();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); 
		glfwSwapBuffers(window);

		auto longend = std::chrono::high_resolution_clock::now();
		longDur = std::chrono::duration_cast<std::chrono::microseconds>(longend - start).count();
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
