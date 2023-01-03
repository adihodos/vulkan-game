use crate::camera::Camera;
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
    position: glm::Vec3,
    view_matrix: glm::Mat4,
    inverse_view: glm::Mat4,
}

impl FlightCamera {
    pub fn new() -> FlightCamera {
        FlightCameraParams::write_default_config();

        FlightCamera {
            params: ron::de::from_reader(
                std::fs::File::open("config/flightcam.cfg.ron")
                    .expect("Failed to read flightcam config"),
            )
            .expect("Invalid flightcam config file."),
            position: glm::Vec3::zeros(),
            view_matrix: glm::Mat4::identity(),
            inverse_view: glm::Mat4::identity(),
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

        self.view_matrix = glm::look_at(&self.position, &look_at, &up_vec);

        let right = self.view_matrix.column(0);
        let up = self.view_matrix.column(1);
        let dir = self.view_matrix.column(2);

        self.inverse_view = glm::Mat4::from_column_slice(&[
            //
            //
            right[0],
            up[0],
            dir[0],
            self.position[0],
            //
            //
            right[1],
            up[1],
            dir[1],
            self.position[1],
            //
            //
            right[2],
            up[2],
            dir[2],
            self.position[2],
            //
            //
            0f32,
            0f32,
            0f32,
            1f32,
        ]);
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
