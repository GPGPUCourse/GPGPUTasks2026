//
// Created by chechevatovr on 15/11/2026.
//

#ifndef GPGPUTASKS_DEBUG_MY_H
#define GPGPUTASKS_DEBUG_MY_H
#include <iostream>

::std::ostream& operator<<(::std::ostream& ostream, const AABBGPU& aabb) {
    ostream << aabb.min_x << ' ' << aabb.min_y << ' ' << aabb.min_z << " -> " << aabb.max_x << ' ' << aabb.max_y << ' ' << aabb.max_z;
    return ostream;
}

::std::ostream& operator<<(::std::ostream& ostream, float3 rhs) {
    ostream << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
    return ostream;
}

#endif // GPGPUTASKS_DEBUG_MY_H
