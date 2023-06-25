use crate::{camera::Camera, math::perspective};
use nalgebra_glm as glm;

#[derive(serde::Serialize, serde::Deserialize)]
struct FlightCameraParams {
    position_relative_to_object: glm::Vec3,
    follow_bias: f32,
    lookahead_factor: f32,
}

impl std::default::Default for FlightCameraParams {
    fn default() -> Self {
        Self {
            position_relative_to_object: glm::vec3(0f32, 1f32, -2f32),
            follow_bias: 0.05f32,
            lookahead_factor: 5f32,
        }
    }
}

impl FlightCameraParams {
    pub fn write_default_config() {
        use ron::ser::{to_writer_pretty, PrettyConfig};

        let cfg_opts = PrettyConfig::new()
            .depth_limit(8)
            .separate_tuple_members(true);

        to_writer_pretty(
            std::fs::File::create("config/flightcam.default.cfg.ron").expect("cykaaaaa"),
            &FlightCameraParams::default(),
            cfg_opts.clone(),
        )
        .expect("Failed to write flight camera config");
    }
}

pub struct FlightCamera {
    params: FlightCameraParams,
    pub position: glm::Vec3,
    pub view_matrix: glm::Mat4,
    pub inverse_view: glm::Mat4,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub fovy: f32,
    pub projection_matrix: glm::Mat4,
    pub inverse_projection: glm::Mat4,
}

impl FlightCamera {
    pub fn new(fovy: f32, aspect: f32, near: f32, far: f32) -> FlightCamera {
        FlightCameraParams::write_default_config();

        let (projection_matrix, inverse_projection) = perspective(fovy, aspect, near, far);

        FlightCamera {
            params: ron::de::from_reader(
                std::fs::File::open("config/flightcam.cfg.ron")
                    .expect("Failed to read flightcam config"),
            )
            .expect("Invalid flightcam config file."),
            position: glm::Vec3::zeros(),
            view_matrix: glm::Mat4::identity(),
            inverse_view: glm::Mat4::identity(),
            near,
            far,
            aspect,
            fovy,
            projection_matrix,
            inverse_projection,
        }
    }

    pub fn update(&mut self, object: &rapier3d::prelude::RigidBody) {
        let ideal_cam_pos = object.position().rotation * self.params.position_relative_to_object
            + object.position().translation.vector;

        let cam_velocity = (ideal_cam_pos - self.position) * self.params.follow_bias;

        self.position += cam_velocity;

        let up_vec = object.position().rotation * glm::Vec3::y_axis();
        let look_at = (object.position().rotation * glm::Vec3::z_axis()).xyz()
            * self.params.lookahead_factor
            + object.position().translation.vector.xyz();

        let view_dir = glm::normalize(&(look_at - self.position));
        let right_dir = glm::normalize(&glm::cross(&view_dir, &up_vec));
        let up_dir = glm::cross(&right_dir, &view_dir);

        let eye_pos = self.position;
        // let look_at = glm::vec3(0f32, 10f32, 10f32);

        self.view_matrix = glm::look_at_lh(&eye_pos, &look_at, &up_dir);
        self.inverse_view = glm::inverse(&self.view_matrix);
        // self.position = eye_pos;

        // self.view_matrix = glm::look_at_lh(&self.position, &look_at, &up_vec);
        // self.inverse_view = glm::inverse(&self.view_matrix);

        // let right = self.view_matrix.column(0);
        // let up = self.view_matrix.column(1);
        // let dir = self.view_matrix.column(2);

        // // log::info!("[{} {} {}]\n{}", right, up, dir, self.view_matrix);

        // self.inverse_view = glm::Mat4::from_column_slice(&[
        //     //
        //     //
        //     right[0],
        //     up[0],
        //     dir[0],
        //     self.position[0],
        //     //
        //     //
        //     right[1],
        //     up[1],
        //     dir[1],
        //     self.position[1],
        //     //
        //     //
        //     right[2],
        //     up[2],
        //     dir[2],
        //     self.position[2],
        //     //
        //     //
        //     0f32,
        //     0f32,
        //     0f32,
        //     1f32,
        // ]);
    }

    pub fn right_up_dir(&self) -> (glm::Vec3, glm::Vec3, glm::Vec3) {
        (
            self.view_matrix.column(0).xyz(),
            self.view_matrix.column(1).xyz(),
            self.view_matrix.column(2).xyz(),
        )
    }
}

impl Camera for FlightCamera {
    fn position(&self) -> glm::Vec3 {
        self.position
    }

    fn view_transform(&self) -> glm::Mat4 {
        self.view_matrix
    }

    fn inverse_view_transform(&self) -> glm::Mat4 {
        self.inverse_view
    }
}
