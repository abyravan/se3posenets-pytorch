#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 vertPosition;
layout (location = 1) in vec3 vertNormal;

out vec3 fragNormal;

void main()
{
   vec4 camPosition = modelViewMatrix*vec4(vertPosition, 1.0);

   gl_Position = projectionMatrix*camPosition;

   fragNormal = normalize(vec3(modelViewMatrix*vec4(vertNormal, 0.0)));
}
