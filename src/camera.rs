use nalgebra_glm as glm;

pub trait Camera {
    fn view_transform(&self) -> glm::Mat4;
    fn position(&self) -> glm::Vec3;
}
