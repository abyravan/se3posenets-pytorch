#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 vertPosition;

out float fragDepth;

void main()
{
   vec4 camPosition = modelViewMatrix*vec4(vertPosition, 1.0);

   gl_Position = projectionMatrix*camPosition;

   fragDepth = camPosition.z;
}
