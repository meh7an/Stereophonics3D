// Position stored inside a helper called from main, unoptimised so the call survives.
#version 450

layout(location = 0) in vec4 in_position;

void emit(vec4 position) {
    gl_Position = position;
}

void main() {
    emit(in_position * 2.0);
}
