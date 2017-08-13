#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 deformedVertPosition;
layout (location = 1) in vec4 canonicalVertPositionWMeshID; 

out vec4 fragVertex;

void main()
{
   gl_Position = projectionMatrix*modelViewMatrix*vec4(deformedVertPosition, 1.0);

   fragVertex = canonicalVertPositionWMeshID;
}
