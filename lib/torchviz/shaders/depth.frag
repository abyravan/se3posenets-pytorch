#version 330

in float fragDepth;

out float outputColor;

void main()
{
   outputColor = fragDepth;
}
