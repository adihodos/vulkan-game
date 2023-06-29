use enumflags2::bitflags;

use crate::{
    flight_cam::FlightCamera,
    fps_camera::FirstPersonCamera,
    math::{PlaneAabbClassification, AABB3},
    plane::Plane,
};
use nalgebra_glm as glm;

#[bitflags]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FrustrumPlane {
    Top = 1 << 0,
    Bottom = 1 << 1,
    Left = 1 << 2,
    Right = 1 << 3,
    Far = 1 << 4,
    Near = 1 << 5,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Frustrum {
    pub top_face: Plane,
    pub bottom_face: Plane,
    pub left_face: Plane,
    pub right_face: Plane,
    pub far_face: Plane,
    pub near_face: Plane,
}

impl Frustrum {
    pub fn from_fpscam(cam: &FirstPersonCamera) -> Frustrum {
        let half_v_side = cam.far * (cam.fovy * 0.5f32).tan();
        let half_h_side = half_v_side * cam.aspect;
        let front_mul_far = cam.look * cam.far;

        use glm::{cross, normalize};

        Frustrum {
            near_face: Plane::from_normal_and_origin(cam.look, cam.position + cam.near * cam.look),
            far_face: Plane::from_normal_and_origin(-cam.look, cam.position + front_mul_far),
            right_face: Plane::from_normal_and_origin(
                normalize(&cross(&(front_mul_far + cam.right * half_h_side), &cam.up)),
                cam.position,
            ),
            left_face: Plane::from_normal_and_origin(
                normalize(&cross(&cam.up, &(front_mul_far - cam.right * half_h_side))),
                cam.position,
            ),
            top_face: Plane::from_normal_and_origin(
                normalize(&cross(&cam.right, &(front_mul_far + cam.up * half_v_side))),
                cam.position,
            ),
            bottom_face: Plane::from_normal_and_origin(
                normalize(&cross(&(front_mul_far - cam.up * half_v_side), &cam.right)),
                cam.position,
            ),
        }
    }

    pub fn from_flight_cam(cam: &FlightCamera) -> Self {
        use glm::{cross, normalize};
        let half_v_side = cam.far * (cam.fovy * 0.5f32).tan();
        let half_h_side = half_v_side * cam.aspect;

        let (right, up, look) = cam.right_up_dir();
        let (right, up, look) = (normalize(&right), normalize(&up), normalize(&look));

        let front_mul_far = look * cam.far;

        Frustrum {
            top_face: Plane::from_normal_and_origin(
                normalize(&cross(&right, &(front_mul_far + up * half_v_side))),
                cam.position,
            ),

            bottom_face: Plane::from_normal_and_origin(
                normalize(&cross(&(front_mul_far - up * half_v_side), &right)),
                cam.position,
            ),

            left_face: Plane::from_normal_and_origin(
                normalize(&cross(&up, &(front_mul_far - right * half_h_side))),
                cam.position,
            ),

            right_face: Plane::from_normal_and_origin(
                normalize(&cross(&(front_mul_far + right * half_h_side), &up)),
                cam.position,
            ),

            far_face: Plane::from_normal_and_origin(-look, cam.position + front_mul_far),

            near_face: Plane::from_normal_and_origin(look, cam.position + cam.near * look),
        }
    }
}

pub fn is_aabb_on_frustrum(
    f: &Frustrum,
    aabb: &AABB3,
    transform: &nalgebra::Isometry3<f32>,
) -> bool {
    let mtx = transform.to_homogeneous();
    let aabb = mtx * (*aabb);

    aabb.classify(&f.top_face) != PlaneAabbClassification::NegativeSide
        && aabb.classify(&f.bottom_face) != PlaneAabbClassification::NegativeSide
        && aabb.classify(&f.near_face) != PlaneAabbClassification::NegativeSide
        && aabb.classify(&f.far_face) != PlaneAabbClassification::NegativeSide
        && aabb.classify(&f.left_face) != PlaneAabbClassification::NegativeSide
        && aabb.classify(&f.right_face) != PlaneAabbClassification::NegativeSide
}
