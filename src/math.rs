use nalgebra_glm as glm;

/// Symmetric perspective projection with reverse depth (1.0 -> 0.0) and
/// Vulkan coordinate space.
pub fn perspective(vertical_fov: f32, aspect_ratio: f32, n: f32, f: f32) -> glm::Mat4 {
    let fov_rad = vertical_fov * 2.0f32 * std::f32::consts::PI / 360.0f32;
    let focal_length = 1.0f32 / (fov_rad / 2.0f32).tan();

    let x = focal_length / aspect_ratio;
    let y = -focal_length;
    let a: f32 = n / (f - n);
    let b: f32 = f * a;

    // clang-format off
    glm::Mat4::from_column_slice(&[
        x, 0.0f32, 0.0f32, 0.0f32, 0.0f32, y, 0.0f32, 0.0f32, 0.0f32, 0.0f32, a, -1.0f32, 0.0f32,
        0.0f32, b, 0.0f32,
    ])

    //   if (inverse)
    //   {
    //       *inverse = glm::mat4{
    //           1/x,  0.0f, 0.0f,  0.0f,
    //           0.0f,  1/y, 0.0f,  0.0f,
    //           0.0f, 0.0f, 0.0f, -1.0f,
    //           0.0f, 0.0f,  1/B,   A/B,
    //       };
    //   }
    //
    // // clang-format on
    // return projection;
}

pub fn perspective2(rmin: f32, rmax: f32, umin: f32, umax: f32, dmin: f32, dmax: f32) -> glm::Mat4 {
    glm::Mat4::from_column_slice(&[
        //
        //
        (2.0 * dmin) / (rmax - rmin),
        0f32,
        0f32,
        0f32,
        //
        //
        0f32,
        (-2.0 * dmin) / (umax - umin),
        0f32,
        0f32,
        //
        //
        -(rmax + rmin) / (rmax - rmin),
        -(umax + umin) / (umax - umin),
        dmin / (dmax - dmin),
        -1f32,
        //
        //
        0f32,
        0f32,
        (dmax * dmin) / (dmax - dmin),
        0f32,
    ])
}

pub fn orthographic(rmin: f32, rmax: f32, umin: f32, umax: f32, dmin: f32, dmax: f32) -> glm::Mat4 {
    glm::Mat4::from_column_slice(&[
        //
        //
        2f32 / (rmax - rmin),
        0f32,
        0f32,
        0f32,
        //
        //
        0f32,
        2f32 / (umax - umin),
        0f32,
        0f32,
        //
        //
        0f32,
        0f32,
        1f32 / (dmax - dmin),
        0f32,
        //
        //
        -(rmax + rmin) / (rmax - rmin),
        -(umax + umin) / (umax - umin),
        -dmin / (dmax - dmin),
        1f32,
    ])
}

#[derive(Copy, Clone, Debug)]
pub struct AABB3 {
    pub min: glm::Vec3,
    pub max: glm::Vec3,
}

impl AABB3 {
    pub fn new(min: glm::Vec3, max: glm::Vec3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> glm::Vec3 {
        self.min + (self.max - self.min) * 0.5f32
    }

    pub fn extents(&self) -> glm::Vec3 {
        self.max - self.min
    }

    pub fn width(&self) -> f32 {
        (self.max.x - self.min.x).abs()
    }

    pub fn height(&self) -> f32 {
        (self.max.y - self.min.y).abs()
    }

    pub fn depth(&self) -> f32 {
        (self.max.z - self.min.z).abs()
    }

    pub fn identity() -> Self {
        Self::new(
            glm::Vec3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX),
            glm::Vec3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN),
        )
    }

    pub fn add_point(&mut self, pt: glm::Vec3) {
        self.min = glm::min2(&self.min, &pt);
        self.max = glm::max2(&self.max, &pt);
    }
}

impl std::ops::Mul<AABB3> for glm::Mat4 {
    type Output = AABB3;
    fn mul(self, rhs: AABB3) -> Self::Output {
        //
        // adapted from
        // https://gist.github.com/cmf028/81e8d3907035640ee0e3fdd69ada543f

        //
        // transform to center + extents representation
        let center = rhs.center();
        let extents = rhs.max - center;

        //
        // transform center
        let t_center = (self * glm::vec4(center.x, center.y, center.z, 1f32)).xyz();
        //
        // transform extents (take maximum)

        let abs_mat = glm::Mat3::from_columns(&[
            glm::abs(&self.column(0).xyz()),
            glm::abs(&self.column(1).xyz()),
            glm::abs(&self.column(2).xyz()),
        ]);

        let t_extents = abs_mat * extents;

        AABB3 {
            min: t_center - t_extents,
            max: t_center + t_extents,
        }
    }
}

impl std::default::Default for AABB3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::convert::From<[f32; 6]> for AABB3 {
    fn from(pts: [f32; 6]) -> Self {
        Self::new(
            glm::Vec3::new(pts[0], pts[1], pts[2]),
            glm::Vec3::new(pts[3], pts[4], pts[5]),
        )
    }
}

pub fn aabb_merge(a: &AABB3, b: &AABB3) -> AABB3 {
    AABB3::new(glm::min2(&a.min, &b.min), glm::max2(&a.max, &b.max))
}
