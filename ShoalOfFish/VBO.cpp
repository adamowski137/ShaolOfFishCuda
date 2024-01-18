#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "VBO.hpp"

VBO::~VBO()
{
	glDeleteBuffers(1, &ID);
}

VBO::VBO()
{
	glGenBuffers(1, &ID);
}

void VBO::bindVBO()
{
	glBindBuffer(GL_ARRAY_BUFFER, ID);
}
