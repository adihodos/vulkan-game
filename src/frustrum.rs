use crate::{
    color_palettes::StdColors, debug_draw_overlay::DebugDrawOverlay, flight_cam::FlightCamera,
    fps_camera::FpsCamera, math::AABB3, plane::Plane,
};
use nalgebra_glm as glm;
use smallvec::SmallVec;

#[derive(Copy, Clone)]
pub struct Frustrum {
    pub top_face: Plane,
    pub bottom_face: Plane,
    pub left_face: Plane,
    pub right_face: Plane,
    pub far_face: Plane,
    pub near_face: Plane,
}

impl Frustrum {
    pub fn draw_cam_frustrum(cam: &FlightCamera, dbg: &mut DebugDrawOverlay) {
        let hnear = 2f32 * (cam.fovy * 0.5f32).tan() * cam.near;
        let wnear = hnear * cam.aspect;

        let hfar = 2f32 * (cam.fovy * 0.5f32).tan() * cam.far;
        let wfar = hfar * cam.aspect;

        let (right, up, dir) = cam.right_up_dir();

        let plane_idx = [0, 1, 1, 2, 2, 3, 3, 0];

        let points_near = [
            cam.position + cam.near * dir + right * wnear * 0.5f32 - up * hnear * 0.5f32,
            cam.position + cam.near * dir + right * wnear * 0.5f32 + up * hnear * 0.5f32,
            cam.position + cam.near * dir - right * wnear * 0.5f32 + up * hnear * 0.5f32,
            cam.position + cam.near * dir - right * wnear * 0.5f32 - up * hnear * 0.5f32,
        ];

        let points_far = [
            cam.position + cam.far * dir + right * wfar * 0.5f32 - up * hfar * 0.5f32,
            cam.position + cam.far * dir + right * wfar * 0.5f32 + up * hfar * 0.5f32,
            cam.position + cam.far * dir - right * wfar * 0.5f32 + up * hfar * 0.5f32,
            cam.position + cam.far * dir - right * wfar * 0.5f32 - up * hfar * 0.5f32,
        ];

        plane_idx.windows(2).for_each(|idx| {
            dbg.add_line(
                points_near[idx[0]],
                points_near[idx[1]],
                StdColors::RED,
                StdColors::RED,
            );
        });

        plane_idx.windows(2).for_each(|idx| {
            dbg.add_line(
                points_far[idx[0]],
                points_far[idx[1]],
                StdColors::RED,
                StdColors::RED,
            );
        });

        [0, 1, 2, 3].iter().for_each(|i| {
            dbg.add_line(cam.position, points_far[*i], StdColors::RED, StdColors::RED);
        });
    }

    pub fn from_fpscam(cam: &FpsCamera) -> Frustrum {
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

    aabb.is_on_or_forward_plane(&f.left_face)
        && aabb.is_on_or_forward_plane(&f.right_face)
        && aabb.is_on_or_forward_plane(&f.top_face)
        && aabb.is_on_or_forward_plane(&f.bottom_face)
        && aabb.is_on_or_forward_plane(&f.near_face)
        && aabb.is_on_or_forward_plane(&f.far_face)
}

#[derive(Copy, Clone)]
pub struct BoundingFrustrum {
    pub origin: glm::Vec3,
    pub orientation: glm::Quat,
    pub right_slope: f32,
    pub left_slope: f32,
    pub top_slope: f32,
    pub bottom_slope: f32,
    pub near: f32,
    pub far: f32,
}

impl BoundingFrustrum {
    pub fn from_projection_matrix(m: &glm::Mat4) -> Self {
        let homogeneous_pts = [
            //
            // right @ far plane
            glm::vec4(1f32, 0f32, 1f32, 1f32),
            //
            // left
            glm::vec4(-1f32, 0f32, 1f32, 1f32),
            //
            // top
            glm::vec4(0f32, -1f32, 1f32, 1f32),
            //
            // bottom
            glm::vec4(0f32, 1f32, 1f32, 1f32),
            //
            // near
            glm::vec4(0f32, 0f32, 0f32, 1f32),
            //
            // far
            glm::vec4(0f32, 0f32, 1f32, 1f32),
        ];

        let inverse = glm::inverse(m);
        let points = homogeneous_pts
            .iter()
            .map(|hpt| inverse * hpt)
            .collect::<SmallVec<[glm::Vec4; 6]>>();

        let points = [
            points[0] * points[0].z.recip(),
            points[1] * points[1].z.recip(),
            points[2] * points[2].z.recip(),
            points[3] * points[3].z.recip(),
            points[4] * points[4].w.recip(),
            points[5] * points[5].w.recip(),
        ];

        BoundingFrustrum {
            origin: glm::Vec3::zeros(),
            orientation: glm::Quat::identity(),
            right_slope: points[0].x,
            left_slope: points[1].x,
            top_slope: points[2].y,
            bottom_slope: points[3].y,
            near: points[4].z,
            far: points[5].z,
        }
    }
}
