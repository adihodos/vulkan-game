use glfw::{Action, MouseButton, WindowEvent};
use glm::{clamp, cross, dot, inverse, length, normalize, Mat3, Quat, Vec4};
use nalgebra::{DualQuaternion, Isometry, UnitQuaternion};
use nalgebra_glm::{IVec2, Mat4, Vec2, Vec3};
use crate::camera::Camera;

use nalgebra_glm as glm;

#[derive(Copy, Clone, Debug)]
pub struct ArcballCamera {
    translation: Mat4,
    center_translation: Mat4,
    rotation: UnitQuaternion<f32>,
    camera: Mat4,
    inv_camera: Mat4,
    zoom_speed: f32,
    inv_screen: Vec2,
    prev_mouse: Vec2,
    is_rotating: bool,
    is_first_rotation: bool,
    // fov: f32,
}



impl ArcballCamera {
    pub const INITIAL_FOV: f32 = 30f32;
    pub const TRANSLATION_FACTOR: f32 = 1f32;

    pub fn new(center: Vec3, zoom_speed: f32, screen: IVec2) -> Self {
        let mut arcball_cam = ArcballCamera {
            translation: Mat4::new_translation(&Vec3::new(0f32, 0f32, -1f32)),
            center_translation: inverse(&Mat4::new_translation(&center)),
            rotation: UnitQuaternion::identity(),
            camera: Mat4::identity(),
            inv_camera: Mat4::identity(),
            zoom_speed,
            inv_screen: Vec2::new(1f32 / screen.x as f32, 1f32 / screen.y as f32),
            prev_mouse: Vec2::zeros(),
            is_rotating: false,
            is_first_rotation: true,
        };

        arcball_cam.update_camera();
        arcball_cam
    }

    pub fn camera(&self) -> Mat4 {
        self.camera
    }

    fn update_camera(&mut self) {
        self.camera = self.translation * self.rotation.to_homogeneous() * self.center_translation;
        self.inv_camera = inverse(&self.camera);
    }

    pub fn update_screen(&mut self, width: i32, height: i32) {
        self.inv_screen = Vec2::new(1f32 / width as f32, 1f32 / height as f32);
    }

    pub fn rotate(&mut self, mouse_pos: Vec2) {
        let mouse_cur = Vec2::new(
            (mouse_pos.x * 2f32 * self.inv_screen.x - 1f32).clamp(-1f32, 1f32),
            (1f32 - 2f32 * mouse_pos.y * self.inv_screen.y).clamp(-1f32, 1f32),
        );

        let mouse_prev = Vec2::new(
            (self.prev_mouse.x * 2f32 * self.inv_screen.x - 1f32).clamp(-1f32, 1f32),
            (1f32 - 2f32 * self.prev_mouse.y * self.inv_screen.y).clamp(-1f32, 1f32),
        );

        let mouse_cur_ball = Self::screen_to_arcball(mouse_cur);
        let mouse_prev_ball = Self::screen_to_arcball(mouse_prev);
        self.rotation = mouse_cur_ball * mouse_prev_ball * self.rotation;

        self.prev_mouse = mouse_pos;
        self.update_camera();
    }

    pub fn end_rotate(&mut self) {
        self.is_rotating = false;
    }

    pub fn pan(&mut self, mouse_delta: Vec2) {
        let zoom_dist = self.translation.m33.abs();
        let delta = Vec4::new(
            mouse_delta.x * self.inv_screen.x,
            -mouse_delta.y * self.inv_screen.y,
            0f32,
            0f32,
        ) * zoom_dist;

        let motion = self.inv_camera * delta;
        self.center_translation = Mat4::new_translation(&motion.xyz()) * self.center_translation;
        self.update_camera();
    }

    pub fn zoom(&mut self, amount: f32, _elapsed: f32) {
        let motion = Vec3::new(0f32, 0f32, amount);
        self.translation = Mat4::new_translation(&(motion * self.zoom_speed)) * self.translation;
        self.update_camera();
    }

    fn screen_to_arcball(p: Vec2) -> UnitQuaternion<f32> {
        let distance = dot(&p, &p);

        if distance <= 1f32 {
            UnitQuaternion::new_normalize(Quat::new(0f32, p.x, p.y, (1f32 - distance).sqrt()))
        } else {
            let unit_p = normalize(&p);
            UnitQuaternion::new_normalize(Quat::new(0f32, unit_p.x, unit_p.y, 0f32))
        }
    }

    pub fn input_event(&mut self, event: &WindowEvent) {
        match *event {
            WindowEvent::MouseButton(mouse_btn, action, _mods) => {
                if mouse_btn == MouseButton::Button3 {
                    if action == Action::Press {
                        self.is_first_rotation = true;
                        self.is_rotating = true;
                    }

                    if action == Action::Release {
                        self.is_rotating = false;
                        self.is_first_rotation = true;
                    }
                }
            }

            WindowEvent::CursorPos(x, y) => {
                if self.is_rotating {
                    if self.is_first_rotation {
                        self.prev_mouse = Vec2::new(x as f32, y as f32);
                        self.is_first_rotation = false;
                    } else {
                        self.rotate(Vec2::new(x as f32, y as f32));
                    }
                }
            }

            WindowEvent::Scroll(x, y) => self.zoom(y as f32, x as f32),

            _ => {}
        }
    }
}

impl Camera for ArcballCamera {
    fn view_transform(&self) -> Mat4 {
        self.camera
    }

    fn position(&self) -> Vec3 {
        Vec3::new(
            self.inv_camera.m14,
            self.inv_camera.m24,
            self.inv_camera.m34,
        )
    }
}
