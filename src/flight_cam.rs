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
    position: std::cell::RefCell<glm::Vec3>,
    view_matrix: std::cell::RefCell<glm::Mat4>,
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
            position: std::cell::RefCell::new(glm::Vec3::zeros()),
            view_matrix: std::cell::RefCell::new(glm::Mat4::identity()),
        }
    }

    pub fn update(&self, object: &rapier3d::prelude::RigidBody) {
        let ideal_cam_pos = object.position().rotation * self.params.position_relative_to_object
            + object.position().translation.vector;

        let cam_velocity = (ideal_cam_pos - *self.position.borrow()) * self.params.follow_bias;

        *self.position.borrow_mut() += cam_velocity;

        let up_vec = object.position().rotation * glm::Vec3::y_axis();
        let look_at = (object.position().rotation * glm::Vec3::z_axis()).xyz()
            * self.params.lookahead_factor
            + object.position().translation.vector.xyz();

        *self.view_matrix.borrow_mut() = glm::look_at(&self.position.borrow(), &look_at, &up_vec);
    }
}

impl Camera for FlightCamera {
    fn position(&self) -> glm::Vec3 {
        *self.position.borrow()
    }

    fn view_transform(&self) -> glm::Mat4 {
        *self.view_matrix.borrow()
    }
}
