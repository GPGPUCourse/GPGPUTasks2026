#version 450

layout(location = 0) in vec3 worldPosition;

layout(push_constant) uniform Projection {
    vec4 rows[4];
} projection;

float projectRow(vec4 row, vec3 p)
{
    return fma(row.x, p.x, fma(row.y, p.y, fma(row.z, p.z, row.w)));
}

void main()
{
    gl_Position = vec4(projectRow(projection.rows[0], worldPosition),
                       projectRow(projection.rows[1], worldPosition),
                       projectRow(projection.rows[2], worldPosition),
                       projectRow(projection.rows[3], worldPosition));
}
