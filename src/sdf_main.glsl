
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
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

vec3 lerp(vec3 a, vec3 b, float v) {
    return a + (b - a) * v;
}

float smin( float a, float b, float k )
{
    k *= 1.0;
    float r = exp2(-a/k) + exp2(-b/k);
    return -k*log2(r);
}

vec4 sdTorso(vec3 p)
{
    float d1 = sdSphere((p - vec3(0.0, 0.25, -0.7)), 0.25);
    float d2 = sdSphere((p - vec3(0.0, 0.53, -0.7)), 0.12);
    
    return vec4(
       smin(d1, d2, 0.1),
       vec3(0.0, 1.0, 0.0)
    );
}

vec4 sdArms(vec3 p) {
    float dl = sdCapsule(p, vec3(0.35, .2, -0.7), vec3(0.25, .4, -0.7), .03);
    float dr = sdCapsule(
      p, 
      lerp(
        vec3(-0.45, .6, -0.7),
        vec3(-0.35, .2, -0.7), 
        (lazycos(iTime * 8.0) + 1.0) / 2.0
      ), 
      vec3(-0.25, .4, -0.7),
      .03
    );
    return vec4(
        min(dl, dr),
        vec3(0.0, 1.0, 0.0)
    );
}

vec4 sdLegs(vec3 p) {
    if (p.x < 0.0) {
      p.x = -p.x; 
    }
    
    float d = sdCapsule(p, vec3(0.07, 0.10, -0.7), vec3(0.082, -0.03, -0.7), .04);
    return vec4(
        d,
        vec3(0.0, 1.0, 0.0)
    );
}

vec4 sdBody(vec3 p) {
    vec4 torso = sdTorso(p);
    vec4 legs  = sdLegs(p);
    vec4 arms  = sdArms(p);
    
    
    if (legs.x < torso.x) {
      return legs;
    }
    if (arms.x < torso.x) {
      return arms;
    }
    return torso;
    
}

vec4 sdEye(vec3 p)
{

    float dw = sdSphere((p - vec3(0.0, 0.5, -.55)), 0.148);
    float dc = sdSphere((p - vec3(0.0, 0.51, -.485)), 0.100);
    float db = sdSphere((p - vec3(0.0, 0.512, -.451)), 0.072);

    if (db < dc) {
      return vec4(
        db,
        vec3(0.0, 0.0, 0.0)
      );
    }
    if (dc < dw) {
      return vec4(
        dc,
        vec3(0.0, 1.0, 1.0)
     );
    }

    return vec4(
       dw,
       vec3(1.0, 1.0, 1.0)
    );
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
    sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
    sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
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


    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));


    vec4 res = raycast(ray_origin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}

