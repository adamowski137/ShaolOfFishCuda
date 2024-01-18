#include "VAO.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
VAO::VAO()
{
    glGenVertexArrays(1, &ID);
}

VAO::~VAO()
{
    glDeleteVertexArrays(1, &ID);
}

void VAO::bindVAO()
{
    glBindVertexArray(ID);
}
