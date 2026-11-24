
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

float smin(float a, float b, float k)
{
    return min(a, b) - k * log2(1.0 + exp2(-abs(a - b) / k));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float radius)
{
    vec3 ab = b - a;
    float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - (a + t * ab)) - radius;
}

float sdTaperedCapsule(vec3 p, vec3 a, vec3 b, float radius_a, float radius_b)
{
    vec3 ab = b - a;
    float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return 0.98 * (length(p - (a + t * ab)) - mix(radius_a, radius_b, t));
}

vec4 sdJacket(vec3 p, float body, float card_sleeve_shape, float free_sleeve_shape, vec3 cuff_p)
{
    float side = abs(p.x);
    float opening_width = mix(0.081, 0.154, smoothstep(0.29, 0.44, p.y));
    float opening = max(0.80 * (side - opening_width), -0.61 - p.z);
    float jacket_top = 0.438 - 0.29 * clamp(side - 0.150, 0.0, 0.155);
    float top_edge = 0.95 * (p.y - jacket_top);

    float panel_fold = sin(45.0 * p.y + 17.0 * side + 310.0 * (side - 0.20) * (side - 0.20));
    float sleeve_fold = sin(71.0 * p.y - 30.0 * p.x + 24.0 * p.z);
    float torso = max(0.70 * (body - 0.017 - 0.0035 * panel_fold), max(0.15 - p.y, top_edge));
    torso = max(torso, -opening);
    float card_sleeve = max(0.76 * (card_sleeve_shape - 0.0034 * sleeve_fold), top_edge);
    float free_sleeve = 0.76 * (free_sleeve_shape - 0.0034 * sleeve_fold);
    float sleeves = min(card_sleeve, free_sleeve);
    float jacket = min(torso, sleeves);

    float collar_bottom = jacket_top - 0.042 + 0.009 * smoothstep(0.165, 0.230, side);
    float collar_h = p.y - collar_bottom;
    float collar_opening_width = opening_width - (p.x > 0.0 ? 0.025 : 0.012);
    float collar_opening = max(0.80 * (side - collar_opening_width), -0.61 - p.z);
    float collar = max(body - 0.030, max(-collar_h, max(top_edge, -collar_opening)));

    vec2 shoulder = vec2(side - 0.269, p.y - 0.376);
    vec2 pad_p = vec2(dot(shoulder, vec2(0.832, -0.555)),
                      dot(shoulder, vec2(0.555, 0.832)));
    vec2 pad_box = abs(pad_p) - vec2(0.076, 0.028);
    float pad_edge = length(max(pad_box, vec2(0.0)))
                   + min(max(pad_box.x, pad_box.y), 0.0) - 0.015;
    float pad_bounds = max(pad_edge, max(-0.75 - p.z, top_edge));
    float rib_phase = pad_p.x + 2.4 * pad_p.y * pad_p.y;
    float rib = 0.5 + 0.5 * cos(6.28318530718 * rib_phase / 0.025);
    float pad_height = (0.006 + 0.010 * rib) * smoothstep(0.0, 0.014, -pad_edge);
    float padding = max(0.58 * (jacket - pad_height), pad_bounds);

    const vec3 leather = vec3(0.043, 0.041, 0.045);
    if (padding < min(jacket, collar)) {
        float stitch = 1.0 - smoothstep(0.0008, 0.0026, abs(pad_edge + 0.007));
        float piping = 1.0 - smoothstep(0.0010, 0.0030, abs(pad_edge + 0.0025));
        vec3 padded_leather = leather * mix(0.59, 1.27, rib);
        return vec4(padding, padded_leather + vec3(0.031 * stitch + 0.016 * piping));
    }

    if (collar < jacket) {
        float seam = min(abs(collar_h - 0.005), abs(top_edge + 0.006));
        float stitch = 1.0 - smoothstep(0.0007, 0.0025, seam);
        vec3 color = leather * 1.12 + vec3(0.037 * stitch);
        if (p.z > -0.55) {
            float snap_radius = length(vec2(side - (p.x > 0.0 ? 0.142 : 0.156), p.y - 0.417));
            float snap = 1.0 - smoothstep(0.0067, 0.0080, snap_radius);
            float inset = 1.0 - smoothstep(0.0047, 0.0059, snap_radius);
            vec3 metal = mix(vec3(0.29, 0.28, 0.31), vec3(0.13, 0.13, 0.15), inset);
            color = mix(color, metal, snap);
        }
        return vec4(collar, color);
    }

    if (sleeves < torso) {
        float fold = 0.91 + 0.13 * sleeve_fold;
        vec3 color = leather * fold;
        if (free_sleeve < card_sleeve) {
            if (cuff_p.y < 0.034) {
                color *= 0.74;
            }
            if (cuff_p.y > 0.021 && cuff_p.y < 0.063 && cuff_p.z > 0.021) {
                float zip = abs(cuff_p.x - 0.012);
                float tooth = step(0.27, fract((cuff_p.y - 0.021) / 0.005));
                float track = 1.0 - smoothstep(0.0013, 0.0025, zip);
                color = mix(color, vec3(0.10 + 0.29 * tooth), track);
            }
        }
        return vec4(sleeves, color);
    }

    vec3 color = leather * (0.95 + 0.12 * panel_fold);
    if (p.y < 0.171) {
        color *= 0.75;
    }
    if (p.z > -0.55 && p.y > 0.161 && p.y < collar_bottom + 0.002) {
        float zipper_track = side - opening_width - 0.007;
        float tape = 1.0 - smoothstep(0.005, 0.010, abs(zipper_track));
        color *= 1.0 - 0.5 * tape;
        if (abs(zipper_track) < 0.0038) {
            float tooth = step(0.24, fract((p.y - 0.161) / 0.0058));
            color = mix(vec3(0.11, 0.11, 0.12), vec3(0.59, 0.59, 0.63), tooth);
        }
        if (p.x > 0.0 && abs(zipper_track) < 0.006 && p.y < 0.178) {
            color = vec3(0.42, 0.43, 0.47);
        }
    }

    return vec4(torso, color);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p, out bool hit_jacket)
{
    float right_leg = sdSphere((p - vec3(0.1, -0.03, -0.7)), 0.05);
    float left_leg = sdSphere((p - vec3(-0.1, -0.03, -0.7)), 0.05);
    float upper_body = sdSphere((p - vec3(0.0, 0.65, -0.7)), 0.25);
    float lower_body = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.33);
    float body = smin(upper_body, lower_body, 0.06);
    float d = smin(min(left_leg, right_leg), body, 0.02);

    const float elbow_radius = 0.029;
    const float elbow_blend = 0.006;
    const vec2 hand_axis = vec2(cos(-1.22), sin(-1.22));
    const vec2 hand_side = vec2(-hand_axis.y, hand_axis.x);
    const vec3 free_shoulder = vec3(0.25, 0.43, -0.68);
    const vec3 free_elbow = free_shoulder + vec3(0.156 * hand_axis, 0.013);
    const vec3 free_wrist = free_elbow + vec3(0.111 * hand_axis, 0.008);
    const vec3 cuff_axis = normalize(free_wrist - free_elbow);
    float free_upper = sdTaperedCapsule(p, free_shoulder, free_elbow, 0.045, elbow_radius);
    float free_forearm = sdTaperedCapsule(p, free_elbow, free_wrist, elbow_radius, 0.037);
    float free_arm = smin(free_upper, free_forearm, elbow_blend);
    float shoulder_height = 0.408 + 1.75 * max(p.x - 0.306, 0.0);
    float shoulder_edge = 0.49 * (p.y - shoulder_height);

    vec3 wrist_offset = p - free_wrist;
    float wrist_side = dot(wrist_offset.xy, hand_side);
    vec3 hand_p = vec3(wrist_side, dot(wrist_offset.xy, hand_axis), 1.24 * wrist_offset.z);
    float hand = 0.80 * sdCapsule(hand_p, vec3(0.0, 0.006, 0.0), vec3(0.0, 0.030, 0.0), 0.026);
    float free_skin = smin(max(free_arm, shoulder_edge), hand, 0.008);
    vec3 cuff_p = vec3(wrist_side, -dot(wrist_offset, cuff_axis), wrist_offset.z);
    float free_sleeve = max(max(free_arm - 0.016, shoulder_edge), 0.017 - cuff_p.y);

    vec3 elbow = vec3(-0.412, 0.298, -0.611);
    vec3 wrist = vec3(-0.526, 0.362, -0.515);
    float upper_arm = sdTaperedCapsule(p, vec3(-0.25, 0.404, -0.68), elbow, 0.045, elbow_radius);
    float forearm = sdTaperedCapsule(p, elbow, wrist, elbow_radius, 0.042);
    float raised_arm = smin(upper_arm, forearm, elbow_blend);

    float palm = sdSphere(p - vec3(-0.539, 0.391, -0.503), 0.039);
    vec3 finger_p = vec3(abs(p.x + 0.539), p.y, p.z);
    float fingers = sdCapsule(finger_p, vec3(0.018, 0.395, -0.481),
                             vec3(0.017, 0.419, -0.470), 0.011);
    raised_arm = smin(raised_arm, min(palm, fingers), 0.008);
    d = smin(d, min(free_skin, raised_arm), 0.01);

    float sleeve_forearm = sdTaperedCapsule(p, elbow, mix(elbow, wrist, 0.61), elbow_radius, 0.042);
    float raised_sleeve = smin(upper_arm, sleeve_forearm, elbow_blend) - 0.016;

    vec3 color = vec3(0.0, 1.0, 0.0);
    float shirt_neckline = 0.343 + 1.6 * p.x * p.x;
    if (p.z > -0.61 && abs(p.x) < 0.16 && p.y > 0.144 && p.y < shirt_neckline) {
        color = vec3(0.025, 0.025, 0.028);
    }

    float x = p.x / 0.13;
    float top = 0.445 + 0.02 * x * x;
    float bottom = 0.375 + 0.09 * x * x;

    if (p.z > -0.55 && abs(x) < 1.0 && p.y > bottom && p.y < top) {
        color = vec3(0.06, 0.015, 0.025);

        if (x > 0.04 && x < 0.55 && p.y > top - 0.038) {
            color = vec3(1.0, 0.95, 0.85);
        }
    }

    vec4 jacket = sdJacket(p, body, raised_sleeve, free_sleeve, cuff_p);
    hit_jacket = jacket.x < d;
    if (hit_jacket) {
        return jacket;
    }
    return vec4(d, color);
}

vec4 sdEye(vec3 p)
{
    vec3 q = p - vec3(0.0, 0.65, -0.45);
    float d = sdSphere(q, 0.15);
    vec3 color = vec3(1.0);
    float r = length(q.xy);

    if (q.z > 0.0) {
        if (r < 0.045) {
            color = vec3(0.02);
        } else if (r < 0.085) {
            color = vec3(0.0, 0.8, 0.95);
        }
    }

    return vec4(d, color);
}

const vec3 monster_offset = vec3(0.0, 0.08, 0.0);
const vec3 glasses_center = vec3(0.0, 0.638, -0.283);

float glassesOutline(vec2 p)
{
    const float corner = 0.039;
    p.y -= 0.008 * (1.0 - smoothstep(0.020, 0.164, abs(p.x)));
    p.x *= 1.0 + 0.72 * clamp(-p.y, 0.0, 0.115);
    vec2 q = abs(p) - vec2(0.164, 0.111) + vec2(corner);
    return 0.88 * (length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - corner);
}

vec4 sdGlasses(vec3 p)
{
    vec3 q = p - glasses_center;
    float brow = smoothstep(0.035, 0.096, q.y);
    float thickness = mix(0.0024, 0.0038, brow);
    vec2 cross_section = vec2(abs(glassesOutline(q.xy)) - thickness, abs(q.z) - 0.0045);
    float rim = 0.91 * (length(max(cross_section, vec2(0.0)))
                        + min(max(cross_section.x, cross_section.y), 0.0));

    vec3 side_p = vec3(abs(p.x), p.y, p.z);
    float hinges = sdCapsule(side_p, vec3(0.154, 0.719, -0.283),
                             vec3(0.174, 0.719, -0.290), 0.0043);
    float temples = sdCapsule(side_p, vec3(0.174, 0.719, -0.290),
                              vec3(0.207, 0.683, -0.634), 0.0034);

    vec3 graphite = vec3(0.041, 0.043, 0.047);
    vec3 metal = vec3(0.116, 0.122, 0.130);
    if (hinges < min(rim, temples)) {
        return vec4(hinges, vec3(0.22, 0.23, 0.24));
    }
    return vec4(min(rim, temples), temples < rim ? graphite : mix(metal, graphite, brow));
}

vec3 shadeGlassesLens(vec3 color, vec3 ray_origin, vec3 ray_direction, float hit_t)
{
    vec3 lens_center = glasses_center + monster_offset;
    float lens_t = (lens_center.z - ray_origin.z) / ray_direction.z;
    if (lens_t <= 0.0 || lens_t >= hit_t) {
        return color;
    }

    vec2 lens_p = (ray_origin + lens_t * ray_direction - lens_center).xy;
    float outline = glassesOutline(lens_p);
    if (outline >= -0.0015) {
        return color;
    }

    float highlight_p = lens_p.y - (0.066 + 0.48 * lens_p.x);
    float highlight = exp(-abs(highlight_p) * 82.0);
    float second_line = exp(-abs(highlight_p + 0.015) * 245.0);
    float fade = 1.0 - smoothstep(0.030, 0.104, abs(lens_p.x + 0.063));
    float reflection = (0.17 * highlight + 0.065 * second_line) * fade;
    return mix(color * vec3(0.956, 0.981, 1.0), vec3(0.62, 0.79, 0.91), reflection);
}

float sdRoundedBox(vec3 p, vec3 half_size, float radius)
{
    vec3 q = abs(p) - half_size + vec3(radius);
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - radius;
}

float sdCylinderZ(vec3 p, float radius, float half_depth)
{
    vec2 q = vec2(length(p.xy) - radius, abs(p.z) - half_depth);
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}

vec3 graphicsCardFanColor(vec2 p)
{
    float radius = length(p);
    float angle = atan(p.y, p.x + 1e-6);
    float sector = fract(7.0 * angle / 6.28318530718 + 11.0 * radius - 0.10 * iTime);
    float blade = smoothstep(0.03, 0.19, sector) * (1.0 - smoothstep(0.63, 0.77, sector));
    float shine = 0.5 + 0.5 * cos(angle - 2.1);
    return mix(vec3(0.018, 0.020, 0.025), vec3(0.17, 0.182, 0.20),
               blade * (0.55 + 0.45 * shine));
}

vec4 sdGraphicsCard(vec3 p)
{
    vec3 q = p - vec3(-0.539, 0.500, -0.506);
    float tilt = radians(12.0);
    q.xy = mat2(cos(tilt), sin(tilt), -sin(tilt), cos(tilt)) * q.xy;

    float housing = sdRoundedBox(q, vec3(0.191, 0.089, 0.030), 0.013);
    float face = sdRoundedBox(q - vec3(0.0, 0.0, 0.028), vec3(0.184, 0.084, 0.009), 0.006);
    vec3 fan_p = vec3(abs(q.x) - 0.098, q.y, q.z);
    float recess = sdCylinderZ(fan_p - vec3(0.0, 0.0, 0.029), 0.075, 0.020);

    vec4 res = vec4(max(housing, -recess), vec3(0.075, 0.080, 0.090));
    float face_distance = max(face, -recess);
    if (face_distance <= res.x) {
        float fan_radius = length(fan_p.xy);
        float bezel = smoothstep(0.075, 0.081, fan_radius);
        float cross_brace = 1.0 - smoothstep(0.004, 0.009, abs(abs(q.x) - 0.84 * abs(q.y)));
        vec3 silver = vec3(0.33, 0.35, 0.38) + vec3(0.12) * cross_brace;
        vec3 color = mix(vec3(0.11, 0.12, 0.135), silver, bezel);
        if (-recess > face) {
            color = vec3(0.047, 0.051, 0.059);
        }
        res = vec4(face_distance, color);
    }

    float fan = sdCylinderZ(fan_p - vec3(0.0, 0.0, 0.016), 0.071, 0.006);
    if (fan < res.x) {
        res = vec4(fan, graphicsCardFanColor(fan_p.xy));
    }
    float hub = sdCylinderZ(fan_p - vec3(0.0, 0.0, 0.024), 0.022, 0.005);
    if (hub < res.x) {
        float brushed = 0.5 + 0.5 * cos(2.0 * atan(fan_p.y, fan_p.x + 1e-6));
        res = vec4(hub, vec3(0.16, 0.17, 0.19) + vec3(0.095) * brushed);
    }

    return res;
}

vec4 sdMonster(vec3 p, bool include_glasses, out bool hit_jacket)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= monster_offset;

    vec4 res = sdBody(p, hit_jacket);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
        hit_jacket = false;
    }

    if (include_glasses) {
        vec4 glasses = sdGlasses(p);
        if (glasses.x < res.x) {
            res = glasses;
            hit_jacket = false;
        }
    }

    vec4 card = sdGraphicsCard(p);
    if (card.x < res.x) {
        res = card;
        hit_jacket = false;
    }

    return res;
}


float monsterDistance(vec3 p)
{
    bool hit_jacket;
    return sdMonster(p, true, hit_jacket).x;
}

vec3 calcNormal(vec3 p)
{
    const float eps = 0.0001;
    const vec2 h = vec2(eps, 0.0);
    return normalize(vec3(
        monsterDistance(p + h.xyy) - monsterDistance(p - h.xyy),
        monsterDistance(p + h.yxy) - monsterDistance(p - h.yxy),
        monsterDistance(p + h.yyx) - monsterDistance(p - h.yyx)
    ));
}

const float no_hit = 1e10;

bool clipToMonster(vec3 origin, vec3 direction, inout float t_min, inout float t_max)
{
    vec3 p = origin - monster_offset;
    const vec3 lower = vec3(-0.795, -0.098, -1.10);
    const vec3 upper = vec3(0.45, 0.94, -0.263);

    for (int axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) < 1e-7) {
            if (p[axis] < lower[axis] || p[axis] > upper[axis]) {
                return false;
            }
        } else {
            float a = (lower[axis] - p[axis]) / direction[axis];
            float b = (upper[axis] - p[axis]) / direction[axis];
            t_min = max(t_min, min(a, b));
            t_max = min(t_max, max(a, b));
            if (t_min > t_max) {
                return false;
            }
        }
    }
    return true;
}

vec4 raycastMonster(vec3 origin, vec3 direction, float t_max, bool include_glasses, out bool hit_jacket)
{
    hit_jacket = false;
    float t = 0.0;
    if (!clipToMonster(origin, direction, t, t_max)) {
        return vec4(no_hit, vec3(0.0));
    }

    for (int iter = 0; iter < 200 && t <= t_max; ++iter) {
        vec4 res = sdMonster(origin + t * direction, include_glasses, hit_jacket);
        t += res.x;
        if (res.x < 0.001 && t <= t_max) {
            return vec4(t, res.yzw);
        }
    }
    return vec4(no_hit, vec3(0.0));
}

float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{
    vec3 light_dir = p - light_source;
    float target_dist = length(light_dir);
    float t_max = target_dist - 0.001;
    bool hit_jacket;
    return raycastMonster(light_source, light_dir / target_dist, t_max, false, hit_jacket).x < t_max
        ? 0.5 : 1.0;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec2 screen = uv - 0.5 * wh - vec2(0.065, 0.0);
    vec3 ray_direction = normalize(vec3(screen, -1.0));


    float floor_t = ray_direction.y < 0.0 ? -ray_origin.y / ray_direction.y : no_hit;
    bool hit_jacket;
    vec4 res = raycastMonster(ray_origin, ray_direction, floor_t, true, hit_jacket);
    bool hit_floor = floor_t < res.x;
    if (hit_floor) {
        res = vec4(floor_t, 1.0, 0.0, 0.0);
    }
    if (res.x >= no_hit) {
        vec3 background = shadeGlassesLens(vec3(0.0), ray_origin, ray_direction, res.x);
        fragColor = vec4(background, 1.0);
        return;
    }



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = hit_floor ? vec3(0.0, 1.0, 0.0) : calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = min(shading(surface_point, light_source, normal),
                     castShadow(surface_point, light_source));
    col *= shad;

    if (hit_jacket && !hit_floor) {
        vec3 reflection = reflect(ray_direction, normal);
        float left_gloss = pow(max(dot(reflection, vec3(-0.80, 0.20, 0.566)), 0.0), 8.0);
        float right_gloss = pow(max(dot(reflection, vec3(0.76, 0.18, 0.624)), 0.0), 10.0);
        col += vec3(0.31, 0.29, 0.35) * left_gloss + vec3(0.29, 0.28, 0.30) * right_gloss;
    } else {
        float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
        col += vec3(spec);
    }



    // Output to screen
    fragColor = vec4(shadeGlassesLens(col, ray_origin, ray_direction, res.x), 1.0);
}