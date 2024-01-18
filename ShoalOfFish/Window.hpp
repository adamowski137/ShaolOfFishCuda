#pragma once
#include "Shader.hpp"
#include <GLFW/glfw3.h>

class Window
{
private:
	static GLFWwindow* window;
	static int HEIGHT;
	static int WIDTH;
	static void setOneTimeShaderData();
	static Shader* shader;
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static float xMouse;
	static float yMouse;
	static bool gpuMode;
	Window();
public:
	static Window getInstance();
	void runWindow();
	~Window();
};