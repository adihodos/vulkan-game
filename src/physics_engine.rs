use crossbeam::channel::Receiver;
use nalgebra_glm as glm;
use rapier3d::prelude::*;

use crate::{debug_draw_overlay::DebugDrawOverlay, game_world::QueuedCommand};

#[derive(Copy, Clone, Debug)]
pub struct ColliderUserData(u128);

impl ColliderUserData {
    pub fn new(body: RigidBodyHandle) -> ColliderUserData {
        let (handle_low, handle_high) = body.into_raw_parts();

        ColliderUserData(((handle_low as u64) | ((handle_high as u64) << 32)) as u128)
    }

    pub fn rigid_body(self) -> RigidBodyHandle {
        let id = (self.0 & 0x00000000_00000000_00000000_FFFFFFFFu128) as u32;
        let gen = ((self.0 & 0x00000000_00000000_FFFFFFFF_00000000u128) >> 32) as u32;
        RigidBodyHandle::from_raw_parts(id, gen)
    }
}

impl std::convert::From<ColliderUserData> for u128 {
    fn from(c: ColliderUserData) -> Self {
        c.0
    }
}

impl std::fmt::Display for ColliderUserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::write!(f, "[ColliderUseData -> {:#?}]", self.rigid_body())
    }
}

#[cfg(test)]
mod tests {
    use super::ColliderUserData;
    use rapier3d::prelude::RigidBodyHandle;

    #[test]
    fn collider_user_data() {
        let rbody = RigidBodyHandle::from_raw_parts(70, 5);
        let cd = ColliderUserData::new(rbody);
        assert_eq!(rbody, cd.rigid_body());
    }
}

pub struct PhysicsObjectGroups {}

impl PhysicsObjectGroups {
    pub const SHIPS: Group = Group::GROUP_1;
    pub const PROJECTILES: Group = Group::GROUP_2;
    pub const MISSILES: Group = Group::GROUP_3;
}

pub struct PhysicsObjectCollisionGroups {}

impl PhysicsObjectCollisionGroups {
    pub fn ships() -> InteractionGroups {
        InteractionGroups::new(
            PhysicsObjectGroups::SHIPS,
            PhysicsObjectGroups::SHIPS
                | PhysicsObjectGroups::MISSILES
                | PhysicsObjectGroups::PROJECTILES,
        )
    }

    pub fn projectiles() -> InteractionGroups {
        InteractionGroups::new(PhysicsObjectGroups::PROJECTILES, PhysicsObjectGroups::SHIPS)
    }

    pub fn missiles() -> InteractionGroups {
        InteractionGroups::new(PhysicsObjectGroups::MISSILES, PhysicsObjectGroups::SHIPS)
    }
}

pub struct PhysicsEngine {
    gravity: glm::Vec3,
    integration_params: IntegrationParameters,
    debug_render_pipeline: DebugRenderPipeline,
    physics_pipeline: PhysicsPipeline,
    query_pipeline: QueryPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    physics_hooks: (),
    event_handler: ChannelEventCollector,
    collision_recv: Receiver<CollisionEvent>,
    contact_force_recv: Receiver<ContactForceEvent>,
}

impl PhysicsEngine {
    pub fn new() -> PhysicsEngine {
        let mut integration_params = IntegrationParameters::default();
        integration_params.dt = 1f32 / 240f32;

        let (collision_send, collision_recv) = crossbeam::channel::unbounded();
        let (contact_force_send, contact_force_recv) = crossbeam::channel::unbounded();
        let event_handler = ChannelEventCollector::new(collision_send, contact_force_send);

        PhysicsEngine {
            gravity: glm::vec3(0f32, 0f32, 0f32),
            integration_params,
            debug_render_pipeline: DebugRenderPipeline::new(
                DebugRenderStyle::default(),
                DebugRenderMode::COLLIDER_SHAPES | DebugRenderMode::RIGID_BODY_AXES,
            ),
            physics_pipeline: PhysicsPipeline::new(),
            query_pipeline: QueryPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            physics_hooks: (),
            event_handler,
            collision_recv,
            contact_force_recv,
        }
    }

    pub fn update(&mut self, cmds: &mut Vec<QueuedCommand>) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
	    None,
            &self.physics_hooks,
            &self.event_handler,
        );

	self.query_pipeline.update(&self.rigid_body_set, &self.collider_set);

        while let Ok(collision_event) = self.collision_recv.try_recv() {
            if !collision_event.sensor() || !collision_event.started() || collision_event.removed()
            {
                continue;
            }

            if let Some(true) = self
                .narrow_phase
                .intersection_pair(collision_event.collider1(), collision_event.collider2())
            {
                self.collider_set
                    .iter()
                    .find(|&(collider_handle, collider)| {
                        (collider_handle == collision_event.collider1()
                            || collider_handle == collision_event.collider2())
                            && collider.is_sensor()
                    })
                    .map(|(_collider_handle, collider)| {
                        //
                        // check what exactly collided (bullet/missile)
                        if collider
                            .collision_groups()
                            .memberships
                            .intersects(PhysicsObjectGroups::MISSILES)
                        {
                        } else if collider
                            .collision_groups()
                            .memberships
                            .intersects(PhysicsObjectGroups::PROJECTILES)
                        {
                            cmds.push(QueuedCommand::ProcessProjectileImpact(
                                collider
                                    .parent()
                                    .expect("Collider does not have a rigid body attached"),
                            ));
                        }
                    });
            }
        }
    }

    pub fn cast_ray(
        &self,
        origin: nalgebra::Point3<f32>,
        direction: glm::Vec3,
        max_toi: f32,
        filter: QueryFilter,
    ) -> Option<(ColliderHandle, Real)> {
        self.query_pipeline.cast_ray(
            &self.rigid_body_set,
            &self.collider_set,
            &Ray::new(origin, direction),
            max_toi,
            true,
            filter,
        )
    }

    pub fn debug_draw(&mut self, backend: &mut DebugDrawOverlay) {
        self.debug_render_pipeline.render(
            backend,
            &self.rigid_body_set,
            &self.collider_set,
            &self.impulse_joint_set,
            &self.multibody_joint_set,
            &self.narrow_phase,
        );
    }

    pub fn get_collider(&self, handle: ColliderHandle) -> &Collider {
        self.collider_set
            .get(handle)
            .expect(&format!("collider {:?} not found", handle))
    }

    pub fn get_rigid_body(&self, handle: RigidBodyHandle) -> &RigidBody {
        self.rigid_body_set
            .get(handle)
            .expect(&format!("rigid body {:?} not found", handle))
    }

    pub fn get_rigid_body_mut(&mut self, handle: RigidBodyHandle) -> &mut RigidBody {
        self.rigid_body_set
            .get_mut(handle)
            .expect(&format!("rigid body {:?} not found", handle))
    }

    pub fn remove_rigid_body(&mut self, rigid_body_handle: RigidBodyHandle) {
        self.rigid_body_set.remove(
            rigid_body_handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );
    }
}
