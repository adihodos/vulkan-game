use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, Vec3};

pub trait Camera {
    fn view_transform(&self) -> Mat4;
    fn position(&self) -> Vec3;
}
