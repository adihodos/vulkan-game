use nalgebra_glm as glm;

use crate::math::AABB3;

#[derive(Copy, Clone)]
pub struct Plane {
    pub normal: glm::Vec3,
    pub offset: f32,
}

impl std::fmt::Display for Plane {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Plane {{ normal: {}, offset: {} }}",
            self.normal, self.offset
        )
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum PointPlaneClass {
    NegativeHalfspace,
    InPlane,
    PositiveHalfspace,
}

impl Plane {
    pub fn from_normal_and_origin(n: glm::Vec3, o: glm::Vec3) -> Plane {
        let normal = glm::normalize(&n);

        Plane {
            normal,
            offset: -glm::dot(&normal, &o),
        }
    }

    pub fn coord_sys(&self) -> (glm::Vec3, glm::Vec3, glm::Vec3) {
        let u = if self.normal.x.abs() >= self.normal.y.abs() {
            let inv_len = (self.normal.x * self.normal.x + self.normal.z * self.normal.z)
                .sqrt()
                .recip();

            glm::Vec3::new(-self.normal.z * inv_len, 0f32, self.normal.x * inv_len)
        } else {
            let inv_len = (self.normal.y * self.normal.y + self.normal.z * self.normal.z)
                .sqrt()
                .recip();
            glm::Vec3::new(0f32, -self.normal.z * inv_len, self.normal.y * inv_len)
        };

        let v = glm::cross(&self.normal, &u);
        (u, v, self.normal)
    }

    pub fn from_points(p0: glm::Vec3, p1: glm::Vec3, p2: glm::Vec3) -> Plane {
        let u = p1 - p0;
        let v = p2 - p0;
        let normal = glm::normalize(&glm::cross(&v, &u));

        Plane {
            normal,
            offset: -glm::dot(&normal, &p0),
        }
    }

    pub fn set(a: f32, b: f32, c: f32, d: f32) -> Plane {
        let len_sqr = a * a + b * b + c * c;
        let k = len_sqr.sqrt().recip();

        Plane {
            normal: glm::make_vec3(&[a * k, b * k, c * k]),
            offset: d * k,
        }
    }

    pub fn signed_distance(&self, p: &glm::Vec3) -> f32 {
        glm::dot(&self.normal, p) + self.offset
    }

    pub fn classify_point(&self, p: &glm::Vec3) -> PointPlaneClass {
        let sign_dst = self.signed_distance(p);
        if sign_dst > 0f32 {
            PointPlaneClass::PositiveHalfspace
        } else if sign_dst < 0f32 {
            PointPlaneClass::NegativeHalfspace
        } else {
            PointPlaneClass::InPlane
        }
    }

    pub fn intersects_aabb(&self, aabb: &AABB3) -> bool {
        let mut dmin = glm::Vec3::zeros();
        let mut dmax = glm::Vec3::zeros();

        for i in 0..3 {
            if self.normal[i] >= 0f32 {
                dmin[i] = aabb.min[i];
                dmax[i] = aabb.max[i];
            } else {
                dmin[i] = aabb.max[i];
                dmax[i] = aabb.min[i];
            }
        }

        if (glm::dot(&self.normal, &dmin) + self.offset) >= 0f32 {
            false
        } else {
            true
        }
    }
}
