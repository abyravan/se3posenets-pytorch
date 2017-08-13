#version 330

const vec3 lightColor = vec3(1,1,1);
const vec3 lightPos = vec3(0,0,0);
const vec3 surfaceColor = vec3(0.5,0.5,0.5);

in vec3 fragPosition;
in vec3 fragNormal;

out vec4 outputColor;


void main()
{
   vec3 surfaceToLight = lightPos - fragPosition;
   float brightness = dot(fragNormal, surfaceToLight) / (length(surfaceToLight));
   brightness = clamp(brightness, 0, 1);
   outputColor = vec4(brightness * lightColor * surfaceColor, 1.0);

   //outputColor = vec4(vec3(0.5) + 0.5*fragNormal, 1.0);
}
