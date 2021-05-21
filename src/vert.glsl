#version 450 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 vel;

out vec4 f_col;

uniform float scale;
uniform vec2 cpos;

const float PI = 3.1415926535897932384626433832795;

float atan2(in float y, in float x)
{
    bool s = (abs(x) > abs(y));
    return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

void main() {
    vec2 r_pos = ((cpos + pos * vec2(9.0/16.0, 1.0) - 0.5) * 2.0) * scale;
    gl_Position = vec4(r_pos, 0.0, 1.0);
    vec3 d = vec3(sqrt(abs(vel)) * 0.05, 1.0);
    float a = atan(vel.y, vel.x) + 1.5 * PI;
    //float v = (1. * a) / (PI * 2.0);
    //float v0 = length(vel) * 0.005;

    f_col = vec4(d, 0.1);
    gl_PointSize = min(10.0, max(0.5, pow(scale, 2.0) * 9.0));
    //f_col = vec4(vec3(1.0), max(0.1, v0));
    float v = clamp(sqrt(length(vel)) * 0.003, 0.0, 1.0);
    //f_col = vec4(0.0, 0.0, v, 1.0);
    f_col = vec4(clamp(vec3(1.0) - abs(3.0 * vec3(v) + vec3(-3, -2, -1.0)),
                  vec3(0.1), vec3(1)) * 0.2, 0.8);
    //f_col = vec3(1.0) - f_col;
}
