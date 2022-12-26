#[derive(Copy, Clone, Debug)]
pub struct GameObjectPhysicsData {
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub collider_handle: rapier3d::prelude::ColliderHandle,
}

#[derive(Copy, Clone, Debug)]
pub struct GameObjectRenderState {
    pub physics_pos: nalgebra::Isometry3<f32>,
    pub render_pos: nalgebra::Isometry3<f32>,
}
