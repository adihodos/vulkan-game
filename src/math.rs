use nalgebra_glm as glm;

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
