#version 450 core
layout(location = 0) in vec2 pos;

out vec2 f_tex_pos;

void main() {
    f_tex_pos = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
