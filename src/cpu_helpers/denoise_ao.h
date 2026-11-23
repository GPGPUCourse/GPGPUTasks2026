#pragma once

#include <libimages/images.h>

#include <algorithm>
#include <string>

inline void saveDenoisedAO(const image32f& ao, const image32i& face_ids, int no_face_id, const std::string& results_dir)
{
    const int width = ao.width();
    const int height = ao.height();
    image8u before(width, height, 1);
    image8u after(width, height, 1);
    auto to_byte = [](float value) {
        return static_cast<unsigned char>(value * 255.0f + 0.5f);
    };

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = ao(y, x);
            before(y, x) = to_byte(value);
            int face_id = face_ids(y, x);

            if (face_id != no_face_id) {
                float sum = 0.0f;
                int count = 0;
                for (int ny = std::max(0, y - 1); ny <= std::min(height - 1, y + 1); ++ny) {
                    for (int nx = std::max(0, x - 1); nx <= std::min(width - 1, x + 1); ++nx) {
                        if (face_ids(ny, nx) == face_id) {
                            sum += ao(ny, nx);
                            ++count;
                        }
                    }
                }
                value = sum / count;
            }
            after(y, x) = to_byte(value);
        }
    }
    before.saveBMP(results_dir + "/ao_before.bmp");
    after.saveBMP(results_dir + "/ao_after.bmp");
}
