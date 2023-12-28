#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Rgba32F {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba32F {
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self as *const _ as *const f32, 4) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self as *mut _ as *mut f32, 4) }
    }
}

impl std::convert::From<Rgba32F> for (u8, u8, u8, u8) {
    fn from(value: Rgba32F) -> Self {
        (
            (value.r * 255f32).clamp(0f32, 255f32) as u8,
            (value.g * 255f32).clamp(0f32, 255f32) as u8,
            (value.b * 255f32).clamp(0f32, 255f32) as u8,
            (value.a * 255f32).clamp(0f32, 255f32) as u8,
        )
    }
}

impl std::convert::From<Rgba32F> for (u8, u8, u8) {
    fn from(value: Rgba32F) -> Self {
        (
            (value.r * 255f32).clamp(0f32, 255f32) as u8,
            (value.g * 255f32).clamp(0f32, 255f32) as u8,
            (value.b * 255f32).clamp(0f32, 255f32) as u8,
        )
    }
}

impl std::convert::From<(u8, u8, u8, u8)> for Rgba32F {
    fn from((r, g, b, a): (u8, u8, u8, u8)) -> Self {
        Self::from([r, g, b, a])
    }
}

impl std::convert::From<(u8, u8, u8)> for Rgba32F {
    fn from(c: (u8, u8, u8)) -> Self {
        Self::from((c.0, c.1, c.2, 255u8))
    }
}

impl std::convert::From<[u8; 4]> for Rgba32F {
    fn from(c: [u8; 4]) -> Self {
        Self::from([
            c[0] as f32 / 255f32,
            c[1] as f32 / 255f32,
            c[2] as f32 / 255f32,
            c[3] as f32 / 255f32,
        ])
    }
}

impl std::convert::From<[u8; 3]> for Rgba32F {
    fn from(c: [u8; 3]) -> Self {
        Self::from([
            c[0] as f32 / 255f32,
            c[1] as f32 / 255f32,
            c[2] as f32 / 255f32,
            1f32,
        ])
    }
}

impl std::convert::From<[f32; 4]> for Rgba32F {
    fn from(value: [f32; 4]) -> Self {
        Self {
            r: value[0],
            g: value[1],
            b: value[2],
            a: value[3],
        }
    }
}

impl std::convert::From<[f32; 3]> for Rgba32F {
    fn from(value: [f32; 3]) -> Self {
        Self {
            r: value[0],
            g: value[1],
            b: value[2],
            a: 1f32,
        }
    }
}

impl std::convert::From<u32> for Rgba32F {
    fn from(c: u32) -> Self {
        let a = ((c >> 24) & 0xFF) as f32 / 255f32;
        let b = ((c >> 16) & 0xFF) as f32 / 255f32;
        let g = ((c >> 8) & 0xFF) as f32 / 255f32;
        let r = (c & 0xFF) as f32 / 255f32;

        Self { r, g, b, a }
    }
}

impl std::convert::From<Rgba32F> for u32 {
    fn from(c: Rgba32F) -> Self {
        // comp-sci soy boy FTW !!!!
        // omg them epic funcshoonals
        let (c, _) = c.as_slice().iter().fold((0u32, 0u32), |(acc, i), c| {
            (acc | ((c * 255f32).ceil() as u32) << i * 8, i + 1)
        });

        c
    }
}

use nalgebra_glm as glm;
use num_traits::Zero;

impl std::convert::From<Rgba32F> for glm::Vec4 {
    fn from(value: Rgba32F) -> Self {
        glm::make_vec4(value.as_slice())
    }
}

impl std::convert::From<Rgba32F> for glm::Vec3 {
    fn from(value: Rgba32F) -> Self {
        glm::make_vec3(value.as_slice())
    }
}

impl std::convert::From<Hsv> for Rgba32F {
    fn from(c: Hsv) -> Self {
        if c.s.is_zero() {
            return Rgba32F {
                r: c.v,
                g: c.v,
                b: c.v,
                a: 1f32,
            };
        }

        let hue = if c.h == 360f32 { 0f32 } else { c.h / 60f32 };

        let int_part = hue.floor() as i32;
        let frac_part = hue - int_part as f32;

        let p = c.v * (1f32 - c.s);
        let q = c.v * (1f32 - (c.s * frac_part));
        let t = c.v * (1f32 - (c.s * (1f32 - frac_part)));

        let color_table: [f32; 6 * 3] = [
            //
            // Case 0
            c.v, t, p, //
            // Case 1
            q, c.v, p, //
            // Case 2
            p, c.v, t, //
            // Case 3
            p, q, c.v, //
            // Case 4
            t, p, c.v, //
            // Case 5
            c.v, p, q,
        ];

        Self {
            r: color_table[int_part as usize * 3 + 0],
            g: color_table[int_part as usize * 3 + 1],
            b: color_table[int_part as usize * 3 + 2],
            a: 1f32,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Hsv {
    pub h: f32,
    pub s: f32,
    pub v: f32,
}

impl std::convert::From<Rgba32F> for Hsv {
    fn from(c: Rgba32F) -> Self {
        let max = c.r.max(c.g.max(c.b));
        let min = c.r.min(c.g.min(c.b));

        let mut hsv = Hsv {
            h: 0f32,
            s: 0f32,
            v: 0f32,
        };

        hsv.v = max;
        if !max.is_zero() {
            hsv.s = (max - min) / max;
        } else {
            hsv.h = std::f32::MAX;
            return hsv;
        }

        let delta = max - min;
        if c.r == max {
            hsv.h = (c.g - c.b) / delta;
        } else if c.g == max {
            hsv.h = 2f32 + (c.b - c.r) / delta;
        } else {
            hsv.h = 4f32 + (c.r - c.g) / delta;
        }

        hsv.h *= 60f32;

        if hsv.h < 0f32 {
            hsv.h += 360f32;
        }

        hsv
    }
}
