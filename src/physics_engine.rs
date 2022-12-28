use nalgebra_glm as glm;
use rapier3d::prelude::*;

use crate::{debug_draw_overlay::DebugDrawOverlay, draw_context::DrawContext};

pub struct PhysicsEngine {
    gravity: glm::Vec3,
    integration_params: IntegrationParameters,
    debug_render_pipeline: DebugRenderPipeline,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    physics_hooks: (),
    event_handler: (),
}

impl PhysicsEngine {
    pub fn new() -> PhysicsEngine {
        let mut integration_params = IntegrationParameters::default();
        integration_params.dt = 1f32 / 240f32;

        PhysicsEngine {
            gravity: glm::vec3(0f32, 0f32, 0f32),
            integration_params,
            debug_render_pipeline: DebugRenderPipeline::new(
                DebugRenderStyle::default(),
                DebugRenderMode::COLLIDER_SHAPES | DebugRenderMode::RIGID_BODY_AXES,
            ),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            physics_hooks: (),
            event_handler: (),
        }
    }

    pub fn update(&mut self) {
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
            &self.physics_hooks,
            &self.event_handler,
        );
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
}
