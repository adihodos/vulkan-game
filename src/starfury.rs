use std::cell::RefCell;

use crate::{
    draw_context::DrawContext,
    game_world::GameObjectHandle,
    math::AABB3,
    physics_engine::PhysicsEngine,
    resource_cache::{GeometryRenderInfo, PbrRenderable, PbrRenderableHandle},
};

use glm::Vec3;
use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};
use serde::{Deserialize, Serialize};
use strum_macros;

#[derive(Copy, Clone, Debug, strum_macros::EnumIter, strum_macros::EnumProperty)]
#[repr(u8)]
enum EngineThrusterId {
    #[strum(props(node_id = "engine.thruster.upper.left.front",))]
    UpperLeftFront,

    #[strum(props(node_id = "engine.thruster.upper.left.up",))]
    UpperLeftUp,

    #[strum(props(node_id = "engine.thruster.upper.left.left",))]
    UpperLeftLeft,

    #[strum(props(node_id = "engine.thruster.upper.left.back"))]
    UpperLeftBack,

    #[strum(props(node_id = "engine.thruster.upper.right.front"))]
    UpperRightFront,

    #[strum(props(node_id = "engine.thruster.upper.right.up"))]
    UpperRightUp,

    #[strum(props(node_id = "engine.thruster.upper.right.right"))]
    UpperRightRight,

    #[strum(props(node_id = "engine.thruster.upper.right.back"))]
    UpperRightBack,

    #[strum(props(node_id = "engine.thruster.lower.left.front"))]
    LowerLeftFront,

    #[strum(props(node_id = "engine.thruster.lower.left.down"))]
    LowerLeftDown,

    #[strum(props(node_id = "engine.thruster.lower.left.left"))]
    LowerLeftLeft,

    #[strum(props(node_id = "engine.thruster.lower.left.back"))]
    LowerLeftBack,

    #[strum(props(node_id = "engine.thruster.lower.right.front"))]
    LowerRightFront,

    #[strum(props(node_id = "engine.thruster.lower.right.down"))]
    LowerRightDown,

    #[strum(props(node_id = "engine.thruster.lower.right.right"))]
    LowerRightRight,

    #[strum(props(node_id = "engine.thruster.lower.right.back"))]
    LowerRightBack,
}

#[derive(Serialize, Deserialize)]
struct FlightModel {
    mass: f32,
    linear_damping: f32,
    angular_damping: f32,
    thruster_force_primary: f32,
    thruster_force_secondary: f32,
    thruster_force_vectors: [glm::Vec3; 16],
}

struct EngineThruster {
    name: String,
    transform: glm::Mat4,
    aabb: AABB3,
}

impl FlightModel {
    fn write_default_config() {
        use ron::ser::{to_writer_pretty, PrettyConfig};

        let cfg_opts = PrettyConfig::new()
            .depth_limit(8)
            .separate_tuple_members(true);

        to_writer_pretty(
            std::fs::File::create("config/starfury.flightmodel.cfg.ron").expect("cykaaaaa"),
            &FlightModel::default(),
            cfg_opts.clone(),
        )
        .expect("Failed to write default flight model config");
    }
}

impl std::default::Default for FlightModel {
    fn default() -> Self {
        Self {
            mass: 23_000f32,
            linear_damping: 1f32,
            angular_damping: 1f32,
            thruster_force_primary: 10_000f32,
            thruster_force_secondary: 2500f32,
            thruster_force_vectors: [
                //
                //upper left
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, -1f32, 0f32),
                glm::vec3(1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
                //
                //upper right
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, -1f32, 0f32),
                glm::vec3(-1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
                //
                // Lower left thruster
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, 1f32, 0f32),
                glm::vec3(1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
                //
                // Lower right thruster
                glm::vec3(0f32, 0f32, -1f32),
                glm::vec3(0f32, 1f32, 0f32),
                glm::vec3(-1f32, 0f32, 0f32),
                glm::vec3(0f32, 0f32, 1f32),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ImpulseType {
    Force(glm::Vec3),
    Torque(glm::Vec3),
}

pub struct Starfury {
    pub renderable: PbrRenderableHandle,
    pub object_handle: GameObjectHandle,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub collider_handle: rapier3d::prelude::ColliderHandle,
    flight_model: FlightModel,
    thrusters: Vec<EngineThruster>,
    impulses: RefCell<Vec<ImpulseType>>,
}

impl Starfury {
    pub fn new(
        renderable: PbrRenderableHandle,
        object_handle: GameObjectHandle,
        physics_engine: &mut PhysicsEngine,
        geometry: &GeometryRenderInfo,
    ) -> Starfury {
        // FlightModel::write_default_config();

        let flight_model: FlightModel = ron::de::from_reader(
            std::fs::File::open("config/starfury.flightmodel.cfg.ron")
                .expect("Failed to read Starfury flight model configuration file."),
        )
        .expect("Invalid configuration file");

        use strum::{EnumProperty, IntoEnumIterator};
        let thrusters = EngineThrusterId::iter()
            .map(|thruster_id| {
                let node = geometry
                    .nodes
                    .iter()
                    .find(|node| node.name == thruster_id.get_str("node_id").unwrap())
                    .unwrap();

                EngineThruster {
                    name: node.name.clone(),
                    transform: node.transform,
                    aabb: node.aabb,
                }
            })
            .collect::<Vec<_>>();

        let body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .translation(glm::vec3(0f32, 0f32, 0f32))
            .linear_damping(flight_model.linear_damping)
            .angular_damping(flight_model.angular_damping)
            .build();

        let bbox_half_extents = geometry.aabb.extents() * 0.5f32;
        let collider = ColliderBuilder::cuboid(
            bbox_half_extents.x,
            bbox_half_extents.y,
            bbox_half_extents.z,
        )
        .mass(flight_model.mass)
        .build();

        let body_handle = physics_engine.rigid_body_set.insert(body);
        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            body_handle,
            &mut physics_engine.rigid_body_set,
        );

        Starfury {
            renderable,
            object_handle,
            rigid_body_handle: body_handle,
            collider_handle,
            flight_model,
            thrusters,
            impulses: RefCell::new(Vec::new()),
        }
    }

    pub fn input_event(&self, event: &winit::event::KeyboardInput) {
        use winit::event::VirtualKeyCode;

        let impulse = event.virtual_keycode.and_then(|key_code| match key_code {
            VirtualKeyCode::W => Some(ImpulseType::Force(
                self.flight_model.thruster_force_primary
                    * self.flight_model.thruster_force_vectors
                        [EngineThrusterId::UpperLeftBack as usize],
            )),
            VirtualKeyCode::S => Some(ImpulseType::Force(
                self.flight_model.thruster_force_primary
                    * self.flight_model.thruster_force_vectors
                        [EngineThrusterId::UpperLeftFront as usize],
            )),
            _ => None,
        });

        impulse.map(|i| {
            self.impulses.borrow_mut().push(i);
        });
    }

    pub fn physics_update(&self, phys_engine: &mut PhysicsEngine) {
        if self.impulses.borrow().is_empty() {
            return;
        }

        let rigid_body = phys_engine
            .rigid_body_set
            .get_mut(self.rigid_body_handle)
            .unwrap();

        self.impulses
            .borrow()
            .iter()
            .for_each(|&impulse| match impulse {
                ImpulseType::Force(f) => {
                    rigid_body.add_force(f, true);
                }

                ImpulseType::Torque(t) => rigid_body.add_torque(t, true),
            });

        self.impulses.borrow_mut().clear();
    }
}
