use crate::camera::Camera;
use nalgebra_glm as glm;

#[derive(serde::Serialize, serde::Deserialize)]
struct FlightCameraParams {
    position_relative_to_object: glm::Vec3,
    follow_bias: f32,
}

impl std::default::Default for FlightCameraParams {
    fn default() -> Self {
        Self {
            position_relative_to_object: glm::vec3(0f32, 1f32, -2f32),
            follow_bias: 0.05f32,
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
    position_relative_to_object: glm::Vec3,
    position: std::cell::RefCell<glm::Vec3>,
    follow_bias: f32,
    view_matrix: std::cell::RefCell<glm::Mat4>,
}

impl FlightCamera {
    pub fn new() -> FlightCamera {
        FlightCameraParams::write_default_config();

        let camera_config: FlightCameraParams = ron::de::from_reader(
            std::fs::File::open("config/flightcam.cfg.ron")
                .expect("Failed to read flightcam config"),
        )
        .expect("Invalid flightcam config file.");

        FlightCamera {
            position_relative_to_object: camera_config.position_relative_to_object,
            position: std::cell::RefCell::new(glm::Vec3::zeros()),
            follow_bias: camera_config.follow_bias,
            view_matrix: std::cell::RefCell::new(glm::Mat4::identity()),
        }
    }

    pub fn update(&self, object_pos: &nalgebra::Isometry3<f32>, delta_tm: f32) {
        let ideal_cam_pos =
            object_pos.rotation * self.position_relative_to_object + object_pos.translation.vector;

        let velocity_vector = ideal_cam_pos - *self.position.borrow();

        // let position = glm::lerp(
        //     &self.position.borrow(),
        //     &ideal_cam_pos,
        //     delta_tm * self.follow_bias,
        // );

        let cam_velocity = velocity_vector * self.follow_bias;

        *self.position.borrow_mut() += cam_velocity;
        // *self.position.borrow_mut() = position;

        let up_vec = object_pos.rotation * glm::Vec3::y_axis();

        *self.view_matrix.borrow_mut() = glm::look_at(
            &self.position.borrow(),
            &object_pos.translation.vector,
            &up_vec,
        );
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
