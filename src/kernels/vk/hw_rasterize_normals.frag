#version 450

// libgpu exposes RGB8 as three R8_UNORM attachments.
layout(location = 0) out float outRed;
layout(location = 1) out float outGreen;
layout(location = 2) out float outBlue;

layout(std430, binding = 0) readonly buffer VerticesIn { float vertices[]; };
layout(std430, binding = 1) readonly buffer FacesIn { uint faces[]; };

#include "mesh_helpers.vk"
#include "normal_color.vk"

void main()
{
    uvec3 face = loadFace(uint(gl_PrimitiveID));
    vec3 v0 = loadVertex(face.x);
    vec3 v1 = loadVertex(face.y);
    vec3 v2 = loadVertex(face.z);
    vec3 rgb = vec3(shadeFragment(v0, v1, v2)) / 255.0;
    outRed = rgb.r;
    outGreen = rgb.g;
    outBlue = rgb.b;
}
