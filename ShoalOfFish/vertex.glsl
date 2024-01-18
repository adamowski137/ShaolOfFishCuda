#version 450 core

layout(location = 0) in float x;
layout(location = 1) in float y;
layout(location = 2) in float vx;
layout(location = 3) in float vy;

out float fangle;

void main()
{
	gl_Position = vec4(x, y, 0.0f, 1.0f);
	fangle = atan(vx, vy);
}