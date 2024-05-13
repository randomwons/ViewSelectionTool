#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in float aColor;
layout (location = 2) in vec3 aOffset;

out float color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out mat4 omodel;

void main(){
    
    vec4 pos = model * vec4(aPos + aOffset, 1.0);
    gl_Position = projection * view * pos;
    color = aColor;
    omodel = model;
}