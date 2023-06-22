use crate::camera::Camera;
use glm::{dot, inverse, normalize, Quat, Vec4};
use nalgebra::UnitQuaternion;
use nalgebra_glm::{IVec2, Mat4, Vec2, Vec3};

use nalgebra_glm as glm;

#[derive(Copy, Clone, Debug)]
pub struct ArcballCamera {
    translation: Mat4,
    center_translation: Mat4,
    rotation: UnitQuaternion<f32>,
    view_transform: Mat4,
    inverse_view_transform: Mat4,
    zoom_speed: f32,
    inv_screen: Vec2,
    prev_mouse: Vec2,
    is_rotating: bool,
    is_first_rotation: bool,
    is_panning: bool,
    is_first_panning: bool,
}

/// Adapted from this: https://github.com/Twinklebear/arcball/blob/master/src/lib.rs
impl ArcballCamera {
    pub const INITIAL_FOV: f32 = 30f32;
    pub const TRANSLATION_FACTOR: f32 = 1f32;

    pub fn new(center: Vec3, zoom_speed: f32, screen: IVec2) -> Self {
        let mut arcball_cam = ArcballCamera {
            translation: Mat4::new_translation(&Vec3::new(0f32, 0f32, -1f32)),
            center_translation: inverse(&Mat4::new_translation(&center)),
            rotation: UnitQuaternion::identity(),
            view_transform: Mat4::identity(),
            inverse_view_transform: Mat4::identity(),
            zoom_speed,
            inv_screen: Vec2::new(1f32 / screen.x as f32, 1f32 / screen.y as f32),
            prev_mouse: Vec2::zeros(),
            is_rotating: false,
            is_first_rotation: true,
            is_panning: false,
            is_first_panning: true,
        };

        arcball_cam.update_camera();
        arcball_cam
    }

    pub fn camera(&self) -> Mat4 {
        self.view_transform
    }

    fn update_camera(&mut self) {
        self.view_transform =
            self.translation * self.rotation.to_homogeneous() * self.center_translation;
        self.inverse_view_transform = inverse(&self.view_transform);
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

    pub fn pan(&mut self, mouse_cur: Vec2) {
        let mouse_delta = mouse_cur - self.prev_mouse;
        let zoom_dist = self.translation.m33.abs();
        let delta = Vec4::new(
            mouse_delta.x * self.inv_screen.x,
            -mouse_delta.y * self.inv_screen.y,
            0f32,
            0f32,
        ) * zoom_dist;

        let motion = self.inverse_view_transform * delta;
        self.center_translation = Mat4::new_translation(&motion.xyz()) * self.center_translation;
        self.prev_mouse = mouse_cur;
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

    pub fn input_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

        match event {
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
                modifiers: _,
            } => {
                // info!("Mouse button down: {:?}", mouse_btn);
                if button == &MouseButton::Middle {
                    if state == &ElementState::Pressed {
                        self.is_first_rotation = true;
                        self.is_rotating = true;
                    }

                    if state == &ElementState::Released {
                        self.is_rotating = false;
                        self.is_first_rotation = true;
                    }
                }

                if button == &MouseButton::Right {
                    if state == &ElementState::Pressed {
                        self.is_panning = true;
                        self.is_first_panning = true;
                    }
                    if state == &ElementState::Released {
                        self.is_panning = false;
                        self.is_first_panning = true;
                    }
                }
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
                modifiers: _,
            } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if self.is_rotating {
                    if self.is_first_rotation {
                        self.prev_mouse = Vec2::new(x as f32, y as f32);
                        self.is_first_rotation = false;
                    } else {
                        self.rotate(Vec2::new(x as f32, y as f32));
                    }
                }

                if self.is_panning {
                    if self.is_first_panning {
                        self.prev_mouse = Vec2::new(x as f32, y as f32);
                        self.is_first_panning = false;
                    } else {
                        self.pan(Vec2::new(x as f32, y as f32));
                    }
                }
            }

            WindowEvent::Resized(new_size) => {
                self.update_screen(new_size.width as i32, new_size.height as i32)
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
                modifiers: _,
            } => match delta {
                MouseScrollDelta::LineDelta(_horizontal, vertical) => {
                    self.zoom(*vertical, 0f32);
                }
                MouseScrollDelta::PixelDelta(amount) => {
                    log::info!("Pixel delta {:?}", amount);
                }
            },

            _ => (),
        }
    }
}

impl Camera for ArcballCamera {
    fn view_transform(&self) -> Mat4 {
        self.view_transform
    }

    fn position(&self) -> Vec3 {
        Vec3::new(
            self.inverse_view_transform.m14,
            self.inverse_view_transform.m24,
            self.inverse_view_transform.m34,
        )
    }

    fn inverse_view_transform(&self) -> glm::Mat4 {
        self.inverse_view_transform
    }
}
