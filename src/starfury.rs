use crate::{
    game_world::GameObjectHandle,
    physics_engine::PhysicsEngine,
    resource_cache::{GeometryRenderInfo, PbrRenderable, PbrRenderableHandle},
};

use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};

pub struct Starfury {
    pub renderable: PbrRenderableHandle,
    pub object_handle: GameObjectHandle,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub collider_handle: rapier3d::prelude::ColliderHandle,
}

impl Starfury {
    pub fn new(
        renderable: PbrRenderableHandle,
        object_handle: GameObjectHandle,
        physics_engine: &mut PhysicsEngine,
        geometry: &GeometryRenderInfo,
    ) -> Starfury {
        let body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .translation(glm::vec3(0f32, 0f32, 0f32))
            .build();

        let bbox_half_extents = geometry.aabb.extents() * 0.5f32;
        let collider = ColliderBuilder::cuboid(
            bbox_half_extents.x,
            bbox_half_extents.y,
            bbox_half_extents.z,
        )
        .mass(23_000f32)
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
        }
    }
}
