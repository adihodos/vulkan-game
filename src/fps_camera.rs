use crate::{camera::Camera, math::perspective};
use nalgebra_glm as glm;

#[derive(Copy, Clone)]
pub struct FirstPersonCamera {
    pub position: glm::Vec3,
    pub right: glm::Vec3,
    pub up: glm::Vec3,
    pub look: glm::Vec3,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub fovy: f32,
    // pub near_window_height: f32,
    // pub far_window_height: f32,
    view_synced: bool,
    pub view_matrix: glm::Mat4,
    pub projection_matrix: glm::Mat4,
}

impl Camera for FirstPersonCamera {
    fn view_transform(&self) -> glm::Mat4 {
        self.view_matrix
    }

    fn position(&self) -> glm::Vec3 {
        self.position
    }

    fn inverse_view_transform(&self) -> glm::Mat4 {
        glm::Mat4::identity()
    }
}

impl FirstPersonCamera {
    pub fn new(fovy: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
        let view_matrix = Self::default_view();

        let right = view_matrix.column(0).xyz();
        let up = view_matrix.column(1).xyz();
        let look = view_matrix.column(2).xyz();
        let position = view_matrix.column(3).xyz();
        log::info!("R {} U {} D {} O {}", right, up, look, position);
        let (projection_matrix, _) = perspective(fovy, aspect, znear, zfar);

        FirstPersonCamera {
            position,
            right,
            up,
            look,
            near: znear,
            far: zfar,
            aspect,
            fovy: fovy.to_radians(),
            view_synced: true,
            view_matrix,
            projection_matrix,
        }
    }

    pub fn walk(&mut self, d: f32) {
        //
        // position += d * look
        self.position += d * self.look;
        self.view_synced = false;
    }

    pub fn strafe(&mut self, d: f32) {
        //
        //
        self.position += d * self.right;
        self.view_synced = false;
    }

    pub fn jump(&mut self, d: f32) {
        self.position += d * self.up;
        self.view_synced = false;
    }

    pub fn pitch(&mut self, angle: f32) {
        let rotation = glm::Mat4::new_rotation(self.right * angle.to_radians());

        self.up = rotation.transform_vector(&self.up);
        self.look = rotation.transform_vector(&self.look);
        self.view_synced = false;
    }

    pub fn yaw(&mut self, angle: f32) {
        let rotation = glm::Mat4::new_rotation(self.up * angle.to_radians());

        self.right = rotation.transform_vector(&self.right);
        self.look = rotation.transform_vector(&self.look);
        self.view_synced = false;
    }

    pub fn update_view_matrix(&mut self) {
        if self.view_synced {
            return;
        }

        let look = glm::normalize(&self.look);
        let up = glm::normalize(&glm::cross(&look, &self.right));
        let right = glm::cross(&up, &look);

        let tx = -glm::dot(&self.position, &right);
        let ty = -glm::dot(&self.position, &up);
        let tz = -glm::dot(&self.position, &look);

        self.up = up;
        self.look = look;
        self.right = right;

        self.view_matrix = glm::Mat4::from_row_slice(&[
            self.right.x,
            self.right.y,
            self.right.z,
            tx,
            self.up.x,
            self.up.y,
            self.up.z,
            ty,
            self.look.x,
            self.look.y,
            self.look.z,
            tz,
            0f32,
            0f32,
            0f32,
            1f32,
        ]);

        self.view_synced = true;
    }

    pub fn set_lens(&mut self, fovy: f32, aspect: f32, znear: f32, zfar: f32) {
        (self.projection_matrix, _) = perspective(fovy, aspect, znear, zfar);
        self.aspect = aspect;
        self.fovy = fovy.to_radians();
        self.near = znear;
        self.far = zfar;
    }

    fn default_view() -> glm::Mat4 {
        let position = glm::Vec3::zeros();
        let look_at = glm::Vec3::z_axis();
        let world_up = glm::Vec3::y_axis();

        let view_matrix = glm::look_at_lh(&position, &look_at, &world_up);
        view_matrix
    }

    pub fn reset(&mut self) {
        let view_matrix = Self::default_view();
        self.position = view_matrix.column(3).xyz();
        self.right = view_matrix.column(0).xyz();
        self.up = view_matrix.column(1).xyz();
        self.look = view_matrix.column(2).xyz();
        self.view_matrix = view_matrix;
        self.view_synced = true;
    }
}
