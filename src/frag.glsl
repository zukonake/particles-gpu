#version 450 core

out vec4 col;

in vec4 f_col;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(length(coord) > 0.5)
        discard;
    col = f_col;
}

