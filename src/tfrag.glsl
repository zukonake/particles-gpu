#version 450 core

out vec4 col;

uniform sampler2D tex;

void main() {
    col = texture(tex, gl_FragCoord.xy / vec2(1600, 900));
}

