use nalgebra::ComplexField;
use nalgebra_glm as glm;

use crate::plane::Plane;

///
/// Symmetric perspective projection with reverse depth (1.0 -> 0.0) and
/// Vulkan coordinate space. (left-hand coord system)
pub fn perspective(vertical_fov: f32, aspect_ratio: f32, n: f32, f: f32) -> (glm::Mat4, glm::Mat4) {
    let fov_rad = vertical_fov * 2.0f32 * std::f32::consts::PI / 360.0f32;
    let focal_length = 1.0f32 / (fov_rad / 2.0f32).tan();

    let x = focal_length / aspect_ratio;
    let y = -focal_length;
    // let a: f32 = -n / (f - n);
    // let b: f32 = (n * f) / (f - n);
    // let a = 1f32 / (n - f);
    // let b = -f / (n - f);

    let a = f / (f - n);
    let b = (-f * n) / (f - n);

    (
        //
        // projection
        glm::Mat4::from_column_slice(&[
            //
            //
            x, 0f32, 0f32, 0f32, //
            //
            0f32, y, 0f32, 0f32, //
            //
            0f32, 0f32, a, 1.0f32, //
            //
            0f32, 0f32, b, 0f32,
        ]),
        //
        // inverse of projection
        glm::Mat4::from_column_slice(&[
            //
            //
            1f32 / x,
            0f32,
            0f32,
            0f32,
            //
            //
            0f32,
            1f32 / y,
            0f32,
            0f32,
            //
            //
            0f32,
            0f32,
            0f32,
            -1f32,
            //
            //
            0f32,
            0f32,
            1f32 / b,
            a / b,
        ]),
    )
}

pub fn world_coords_to_screen_coords(
    p: nalgebra::Point3<f32>,
    projection_view_matrix: &glm::Mat4,
    screen_width: f32,
    screen_height: f32,
) -> glm::Vec2 {
    let clip_space_pos = projection_view_matrix * p.to_homogeneous();
    let ndc_pos = clip_space_pos.xyz() / clip_space_pos.w;
    glm::vec2(
        ((ndc_pos.x + 1f32) * 0.5f32) * screen_width,
        ((ndc_pos.y + 1f32) * 0.5f32) * screen_height,
    )
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

impl std::fmt::Display for AABB3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AABB3 {{ {:?} {:?} }}", self.min, self.max)
    }
}

impl AABB3 {
    pub fn new(min: glm::Vec3, max: glm::Vec3) -> Self {
        Self { min, max }
    }

    pub fn unit() -> Self {
        Self::new(
            glm::vec3(-0.5f32, -0.5f32, -0.5f32),
            glm::vec3(0.5f32, 0.5f32, 0.5f32),
        )
    }

    pub fn center(&self) -> glm::Vec3 {
        // self.min + (self.max - self.min) * 0.5f32
        (self.max + self.min) * 0.5f32
    }

    pub fn extents(&self) -> glm::Vec3 {
        let c = self.center();
        self.max - c
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

    pub fn is_on_or_forward_plane(&self, p: &Plane) -> bool {
        let c = self.center();
        let h = (self.max - self.min) * 0.5f32;
        let e = glm::dot(&h, &p.normal.abs());
        let s = p.signed_distance(&c);

        (s - e) >= 0f32
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

pub fn orthonormal_basis_from_vec(n: &glm::Vec3) -> (glm::Vec3, glm::Vec3, glm::Vec3) {
    let axis2 = glm::normalize(n);

    let a = if axis2.x.abs() > 0.9f32 {
        glm::Vec3::new(0f32, 1f32, 0f32)
    } else {
        glm::Vec3::new(1f32, 0f32, 0f32)
    };

    let axis1 = glm::normalize(&glm::cross(&axis2, &a));
    let axis0 = glm::cross(&axis2, &axis1);

    (axis0, axis1, axis2)
}
