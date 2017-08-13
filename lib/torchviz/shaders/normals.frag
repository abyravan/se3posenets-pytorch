#version 330

in vec3 fragNormal;

out vec3 outputColor;

void main()
{
   outputColor = fragNormal;
}
