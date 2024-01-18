#version 450 core

layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

in float fangle[];

void main() {
    float cosAngle = cos(fangle[0]);
    float sinAngle = sin(fangle[0]);
    mat2 rotationMatrix = mat2(cosAngle, -sinAngle, sinAngle, cosAngle);
    vec2 offset = rotationMatrix * vec2(0.003, -0.003);
    gl_Position = (gl_in[0].gl_Position + vec4(offset, 0.0, 0.0));
    EmitVertex();
    offset = rotationMatrix * vec2(-0.003, -0.003);
    gl_Position = (gl_in[0].gl_Position + vec4(offset, 0.0, 0.0));
    EmitVertex();
    offset = rotationMatrix * vec2(0, 0.008);
    gl_Position = (gl_in[0].gl_Position + vec4(offset, 0.0, 0.0));
    EmitVertex();

    EndPrimitive();
}