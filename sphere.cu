#include "sphere.h"

__device__ sphere::sphere() {

}

__device__ sphere::sphere(vec3 center, float radius) {
    this->center = center;
    this->radius = radius;
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        //from the textbook formulala

    // (A - C) , point/origin difference from center of sphere
    vec3 distance = r.origin() - center;

    // vector form t*t*dot(B,B) + 2*t*dot(A-C,A-C) + dot(C,C) - R*R = 0
    // to the form at^2 + bt + c

    // dot(B,B)
    float a = dot_product(r.direction(),r.direction());

    //2*dot(A-C,A-C)
    float b = 2.0f * dot_product( distance, r.direction());

    // dot(A-C,A-C) - R^2
    float c = dot_product(distance, distance) - radius*radius;

    //disriminant (b^2 - 4ac):
    // < 0 no solutions
    // == 0 one solution
    // > 1 two solutions
    float discriminant = b*b  - 4*a*c;

    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/ (2.0f*a);
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = r.point_at_parameter(rec.t);
            rec.normal = (rec.point - center) / radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / (2.0f *a);
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = r.point_at_parameter(rec.t);
            rec.normal = (rec.point - center) / radius;
            return true;
        }
    }
    
    return false;
    

}


__device__ vec3 sphere::getCenter() {
    return this->center;
}

__device__ float sphere::getRadius() {
    return this->radius;
}

__device__ void sphere::setCenter(vec3 newCenter) {
    this->center = newCenter;
}

__device__ void sphere::setRadius(float newRadius) {
    this->radius = newRadius;
}