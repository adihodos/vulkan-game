use nalgebra_glm as glm;

use crate::{
    game_object::GameObjectPhysicsData,
    math::AABB3,
    physics_engine::PhysicsEngine,
    resource_system::{MeshId, ResourceSystem},
};

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShadowFighterSwarmParams {
    pub instance_count: u32,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub position: glm::Vec3,
    pub orientation: glm::Vec3,
}

impl std::default::Default for ShadowFighterSwarmParams {
    fn default() -> Self {
        Self {
            instance_count: 10,
            mass: 40_000f32,
            linear_damping: 1f32,
            angular_damping: 1f32,
            position: glm::vec3(0f32, 10f32, 10f32),
            orientation: glm::Vec3::zeros(),
        }
    }
}

pub struct ShadowFighterSwarm {
    pub mesh_id: MeshId,
    pub bounds: AABB3,
    pub params: ShadowFighterSwarmParams,
    pub instances_physics_data: Vec<GameObjectPhysicsData>,
}

impl ShadowFighterSwarm {
    pub fn new(physics_engine: &mut PhysicsEngine, rsys: &ResourceSystem) -> ShadowFighterSwarm {
        let swarm_params: ShadowFighterSwarmParams = ron::de::from_reader(
            std::fs::File::open("config/shadow.swarm.cfg.ron")
                .expect("Failed to read shadow swarm config"),
        )
        .expect("Invalid shadow swam config file.");

        let mesh_id: MeshId = "shadow.fighter".into();
        let mesh = rsys.get_mesh_info(mesh_id);
        let aabb = mesh.bounds;

        let instances_physics_data = (0..swarm_params.instance_count)
            .map(|instance_id| {
                let position = swarm_params.position + aabb.extents() * instance_id as f32;

                let body = rapier3d::prelude::RigidBodyBuilder::new(
                    rapier3d::prelude::RigidBodyType::Dynamic,
                )
                .translation(position)
                .rotation(swarm_params.orientation)
                .linear_damping(swarm_params.linear_damping)
                .angular_damping(swarm_params.angular_damping)
                .build();

                let bbox_half_extents = aabb.extents();

                let collider = rapier3d::prelude::ColliderBuilder::cuboid(
                    bbox_half_extents.x,
                    bbox_half_extents.y,
                    bbox_half_extents.z,
                )
                .mass(swarm_params.mass)
                .build();

                let body_handle = physics_engine.rigid_body_set.insert(body);
                let collider_handle = physics_engine.collider_set.insert_with_parent(
                    collider,
                    body_handle,
                    &mut physics_engine.rigid_body_set,
                );

                GameObjectPhysicsData {
                    rigid_body_handle: body_handle,
                    collider_handle,
                }
            })
            .collect::<Vec<_>>();

        ShadowFighterSwarm {
            mesh_id,
            bounds: aabb,
            params: swarm_params,
            instances_physics_data,
        }
    }

    pub fn instances(&self) -> &[GameObjectPhysicsData] {
        &self.instances_physics_data
    }
}
