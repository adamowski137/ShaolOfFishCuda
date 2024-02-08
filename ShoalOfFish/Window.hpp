#pragma once
#include "Shader.hpp"
#include <GLFW/glfw3.h>
#include <vector>
#include "FishBase.cuh"

class Window
{
private:
	static GLFWwindow* window;
	static int selectedFish;
	static int HEIGHT;
	static int WIDTH;
	static void setOneTimeShaderData();
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static float xMouse;
	static float yMouse;
	static bool gpuMode;
	static bool mouseClicked;
	static void showImgui(std::vector<FishBase*> fish);
	Window();
public:
	static Window getInstance();
	void runWindow();
	~Window();
};