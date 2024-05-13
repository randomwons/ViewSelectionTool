#version 330 core

in float color;
in mat4 amodel;
out vec4 fragColor;

void main(){
    
    float a = color;
    if(color < 0.3) {
        a = 0.0;
    }

    const float n = 4.0;
    float v = max(0.0, min(n, color * n));
    fragColor = vec4(
        min(max(1.5 - abs(v - 3.0), 0.0), 1.0),
        min(max(1.5 - abs(v - 2.0), 0.0), 1.0),
        min(max(1.5 - abs(v - 1.0), 0.0), 1.0),
        a * 0.1
    );
    gl_FragDepth = amodel[3][1];
}