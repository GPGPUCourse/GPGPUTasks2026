#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

#include <libbase/point.h>

// Save triangle mesh to ASCII PLY.
// - vertices: array of 3D points
// - triangles: each point3u stores vertex indices (x, y, z)
// Returns true on success.
inline bool save_ply_triangles_ascii(
    const std::string& filename,
    const std::vector<point3f>& vertices,
    const std::vector<point3u>& triangles)
{
    const std::uint32_t vertex_count =
        static_cast<std::uint32_t>(vertices.size());
    const std::uint32_t face_count =
        static_cast<std::uint32_t>(triangles.size());

    std::ofstream out(filename.c_str());
    if (!out.is_open())
        return false;

    // Header
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << vertex_count << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "element face " << face_count << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    // Vertices
    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        const point3f& v = vertices[i];
        out << v.x << ' ' << v.y << ' ' << v.z << '\n';
    }

    // Faces (triangles)
    for (std::uint32_t i = 0; i < face_count; ++i) {
        const point3u& f = triangles[i];
        out << "3 " << f.x << ' ' << f.y << ' ' << f.z << '\n';
    }

    return true;
}
