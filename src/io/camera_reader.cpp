#include "camera_reader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cctype>
#include <algorithm>

#include "libutils/misc.h" // rassert
#include <iomanip>

// -------------------- tiny XML helpers --------------------

static std::string findTagOpenToGT(const std::string& xml, const std::string& tagName) {
    const std::string open = "<" + tagName;
    size_t p = xml.find(open);
    rassert(p != std::string::npos, 5101, tagName);
    size_t q = xml.find('>', p);
    rassert(q != std::string::npos, 5102, tagName);
    return xml.substr(p, q - p + 1);
}

static std::string getAttr(const std::string& tag, const std::string& name) {
    const std::string key = name + "=\"";
    size_t p = tag.find(key);
    rassert(p != std::string::npos, 5201, name);
    p += key.size();
    size_t q = tag.find('"', p);
    rassert(q != std::string::npos && q > p, 5202, name);
    return tag.substr(p, q - p);
}

template <class T>
static std::vector<T> parseList(const std::string& s) {
    std::istringstream ss(s);
    std::vector<T> out;
    T v{};
    while (ss >> v) out.push_back(v);
    return out;
}

// -------------------- math helpers --------------------

// row-major 3x3 * vec3
static inline void mul3x3_vec3(const float R[9], const float v[3], float out[3]) {
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

// t = -R * C (our convention: X_cam = R * X_world + t), R is row-major world→camera
static inline void compute_t_from_R_C(const float R[9], const float C[3], float t[3]) {
    float Rc[3];
    mul3x3_vec3(R, C, Rc);
    t[0] = -Rc[0];
    t[1] = -Rc[1];
    t[2] = -Rc[2];
}

// -------------------- core --------------------

CameraViewGPU parseViewStateFromString(const std::string& xml)
{
    const std::string camTag  = findTagOpenToGT(xml, "VCGCamera");
    const std::string viewTag = findTagOpenToGT(xml, "ViewSettings");

    CameraViewGPU out = {};
    out.magic_bits_guard = CAMERA_VIEW_GPU_MAGIC_BITS_GUARD;

    // ---- Intrinsics ----
    {
        const float focal_mm = std::stof(getAttr(camTag, "FocalMm"));
        const auto pix_mm    = parseList<float>(getAttr(camTag, "PixelSizeMm"));   // sx sy  [mm/px]
        const auto center    = parseList<float>(getAttr(camTag, "CenterPx"));      // cx cy  [px]
        const auto viewport  = parseList<uint32_t>(getAttr(camTag, "ViewportPx")); // w h    [px]

        rassert(pix_mm.size() == 2, 5401);
        rassert(center.size() == 2, 5402);
        rassert(viewport.size() == 2, 5403);
        rassert(pix_mm[0] != 0.0f && pix_mm[1] != 0.0f, 5404);

        out.K.focal_mm = focal_mm;
        out.K.pixel_size_mm[0] = pix_mm[0];
        out.K.pixel_size_mm[1] = pix_mm[1];

        out.K.fx = focal_mm / pix_mm[0];   // [px]
        out.K.fy = focal_mm / pix_mm[1];   // [px]
        out.K.cx = center[0];
        out.K.cy = center[1];
        out.K.width  = viewport[0];
        out.K.height = viewport[1];
    }

    // --- Extrinsics ---
    // parse R_wc (row-major), could be 3x3 (9) or 4x4 (16: take upper-left 3x3)
    const auto Rlist = parseList<float>(getAttr(camTag, "RotationMatrix"));
    rassert(Rlist.size() == 9 || Rlist.size() == 16, 5501);
    float R_wc[9];
    if (Rlist.size() == 16) {
#if 0
        // row-major
        R_wc[0]=Rlist[0]; R_wc[1]=Rlist[1]; R_wc[2]=Rlist[2];
        R_wc[3]=Rlist[4]; R_wc[4]=Rlist[5]; R_wc[5]=Rlist[6];
        R_wc[6]=Rlist[8]; R_wc[7]=Rlist[9]; R_wc[8]=Rlist[10];
#else
        // column-major OpenGL
        R_wc[0]=Rlist[0]; R_wc[1]=Rlist[4]; R_wc[2]=Rlist[8];
        R_wc[3]=Rlist[1]; R_wc[4]=Rlist[5]; R_wc[5]=Rlist[9];
        R_wc[6]=Rlist[2]; R_wc[7]=Rlist[6]; R_wc[8]=Rlist[10];
#endif
    } else {
        for (int i=0;i<9;++i) R_wc[i] = Rlist[(size_t)i];
    }

    // forward = третий столбец R_wc (куда «смотрит» камера в мире)
    float fwd[3] = { R_wc[2], R_wc[5], R_wc[8] }; // после фикса индексов выше

    // R_cw = R_wc^T  (store to out.E.R, row-major)
    out.E.R[0]=R_wc[0]; out.E.R[1]=R_wc[3]; out.E.R[2]=R_wc[6];
    out.E.R[3]=R_wc[1]; out.E.R[4]=R_wc[4]; out.E.R[5]=R_wc[7];
    out.E.R[6]=R_wc[2]; out.E.R[7]=R_wc[5]; out.E.R[8]=R_wc[8];

#if 1
    // C = TranslationVector (camera center in world coords)
    const auto Tlist = parseList<float>(getAttr(camTag, "TranslationVector"));
    rassert(Tlist.size() >= 3, 5502);
    out.E.C[0] = -Tlist[0];
    out.E.C[1] = -Tlist[1];
    out.E.C[2] = -Tlist[2];

    // t = -R * C
    out.E.t[0] = -(out.E.R[0]*out.E.C[0] + out.E.R[1]*out.E.C[1] + out.E.R[2]*out.E.C[2]);
    out.E.t[1] = -(out.E.R[3]*out.E.C[0] + out.E.R[4]*out.E.C[1] + out.E.R[5]*out.E.C[2]);
    out.E.t[2] = -(out.E.R[6]*out.E.C[0] + out.E.R[7]*out.E.C[1] + out.E.R[8]*out.E.C[2]);
#else
    const auto Tlist = parseList<float>(getAttr(camTag, "TranslationVector"));
    out.E.t[0] = Tlist[0];
    out.E.t[1] = Tlist[1];
    out.E.t[2] = Tlist[2];

    const float *R = out.E.R;
    out.E.C[0] = -(R[0]*out.E.t[0] + R[3]*out.E.t[1] + R[6]*out.E.t[2]);
    out.E.C[1] = -(R[1]*out.E.t[0] + R[4]*out.E.t[1] + R[7]*out.E.t[2]);
    out.E.C[2] = -(R[2]*out.E.t[0] + R[5]*out.E.t[1] + R[8]*out.E.t[2]);
#endif

    // optional: sanity check, C == -R^T * t
    {
        float Cx = -(out.E.R[0]*out.E.t[0] + out.E.R[3]*out.E.t[1] + out.E.R[6]*out.E.t[2]);
        float Cy = -(out.E.R[1]*out.E.t[0] + out.E.R[4]*out.E.t[1] + out.E.R[7]*out.E.t[2]);
        float Cz = -(out.E.R[2]*out.E.t[0] + out.E.R[5]*out.E.t[1] + out.E.R[8]*out.E.t[2]);
        //rassert(fabs(Cx-out.E.C[0])<1e-3f && fabs(Cy-out.E.C[1])<1e-3f && fabs(Cz-out.E.C[2])<1e-3f, 5510);
    }

    // ---- View settings ----
    {
        out.view.near_plane  = std::stof(getAttr(viewTag, "NearPlane"));
        out.view.far_plane   = std::stof(getAttr(viewTag, "FarPlane"));
        out.view.track_scale = std::stof(getAttr(viewTag, "TrackScale"));
    }

    return out;
}

CameraViewGPU loadViewState(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    rassert(in.good(), 5000, path);
    std::string xml((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parseViewStateFromString(xml);
}

std::string dumpViewStateToString(const CameraViewGPU& camera)
{
    // Reconstruct R_wc (row-major) from stored R_cw = camera.E.R
    float Rwc[9];
    Rwc[0] = camera.E.R[0]; Rwc[1] = camera.E.R[3]; Rwc[2] = camera.E.R[6];
    Rwc[3] = camera.E.R[1]; Rwc[4] = camera.E.R[4]; Rwc[5] = camera.E.R[7];
    Rwc[6] = camera.E.R[2]; Rwc[7] = camera.E.R[5]; Rwc[8] = camera.E.R[8];

    // Prefer given C, but if it looks unset, derive it from t: C = -R^T * t
    float Cx = camera.E.C[0], Cy = camera.E.C[1], Cz = camera.E.C[2];
    const bool c_all_zero = (Cx==0.0f && Cy==0.0f && Cz==0.0f &&
        (camera.E.t[0]!=0.0f || camera.E.t[1]!=0.0f || camera.E.t[2]!=0.0f));
    if (c_all_zero) {
        // R_cw = camera.E.R, so R_cw^T rows are its columns
        const float *R = camera.E.R;
        Cx = -(R[0]*camera.E.t[0] + R[3]*camera.E.t[1] + R[6]*camera.E.t[2]);
        Cy = -(R[1]*camera.E.t[0] + R[4]*camera.E.t[1] + R[7]*camera.E.t[2]);
        Cz = -(R[2]*camera.E.t[0] + R[5]*camera.E.t[1] + R[8]*camera.E.t[2]);
    }

    // Build 4x4 RotationMatrix
#if 0
    // (row-major), upper-left 3x3 = R_wc
    float M[16] = {
        Rwc[0], Rwc[1], Rwc[2], 0.0f,
        Rwc[3], Rwc[4], Rwc[5], 0.0f,
        Rwc[6], Rwc[7], Rwc[8], 0.0f,
        0.0f,   0.0f,   0.0f,   1.0f
    };
        R_wc[0]=Rlist[0]; R_wc[1]=Rlist[1]; R_wc[2]=Rlist[2];
        R_wc[3]=Rlist[4]; R_wc[4]=Rlist[5]; R_wc[5]=Rlist[6];
        R_wc[6]=Rlist[8]; R_wc[7]=Rlist[9]; R_wc[8]=Rlist[10];
#else
    // column-major OpenGL
    float M[16] = {
        Rwc[0], Rwc[3], Rwc[6], 0.0f,
        Rwc[1], Rwc[4], Rwc[7], 0.0f,
        Rwc[2], Rwc[5], Rwc[8], 0.0f,
        0.0f,   0.0f,   0.0f,   1.0f
    };
#endif

    std::ostringstream os;
    // Use "C" locale to ensure dot as decimal separator
    os.imbue(std::locale::classic());

    os << "<!DOCTYPE ViewState>\n";
    os << "<project>\n";
    os << " <VCGCamera";

    // Fixed fields
    os << " LensDistortion=\"0 0\"";
    os << " BinaryData=\"0\"";

    // Intrinsics
    os << " FocalMm=\"" << camera.K.focal_mm << "\"";

    // RotationMatrix
    os << " RotationMatrix=\"";
    for (int i = 0; i < 16; ++i) {
        os << M[i];
        if (i + 1 < 16) os << ' ';
        else os << '"';
    }

    // CenterPx
    os << " CenterPx=\"" << camera.K.cx << ' ' << camera.K.cy << "\"";

    // TranslationVector contains camera center in world coords + homogeneous 1
    os << " TranslationVector=\"" << -Cx << ' ' << -Cy << ' ' << -Cz << " 1\"";

    // Viewport
    os << " ViewportPx=\"" << camera.K.width << ' ' << camera.K.height << "\"";

    // Other fixed fields
    os << " CameraType=\"0\"";

    // Pixel size
    os << " PixelSizeMm=\"" << camera.K.pixel_size_mm[0] << ' ' << camera.K.pixel_size_mm[1] << "\"";

    os << "/>\n";

    // View settings
    os << " <ViewSettings"
       << " FarPlane=\""  << camera.view.far_plane  << "\""
       << " NearPlane=\"" << camera.view.near_plane << "\""
       << " TrackScale=\"" << camera.view.track_scale << "\"/>\n";

    os << "</project>\n";

    return os.str();
}