use std::collections::HashMap;

use crate::{
    draw_context::{DrawContext, InitContext, UpdateContext},
    drawing_system::DrawingSys,
    math::AABB3,
    physics_engine::{PhysicsEngine, PhysicsObjectCollisionGroups},
    resource_system::{EffectType, MeshId, SubmeshId},
};
use nalgebra_glm as glm;
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};

#[derive(
    Copy,
    Clone,
    Debug,
    strum_macros::EnumProperty,
    strum_macros::EnumIter,
    strum_macros::Display,
    Hash,
    Eq,
    PartialEq,
)]
#[repr(u8)]
pub enum MissileKind {
    #[strum(props(kind = "R73"))]
    R73,
    #[strum(props(kind = "R27"))]
    R27,
}

struct MissileClassSheet {
    id: SubmeshId,
    mass: f32,
    aabb: AABB3,
    booster_life: f32,
    thrust: f32,
}

struct SmokePoint {
    pos: glm::Vec3,
    color: glm::Vec4,
}

pub struct Missile {
    kind: MissileKind,
    orientation: nalgebra::Isometry3<f32>,
    rigid_body: RigidBodyHandle,
    collider: ColliderHandle,
    booster_time: f32,
    thrust: f32,
    out_of_vis_range: bool,
    // trail: std::collections::VecDeque<SmokePoint>,
}

impl Missile {
    // fn add_trail_segment(&mut self, current_orientation: &nalgebra::Isometry3<f32>) {
    //     if self.trail.len() > 64 {
    //         self.trail.pop_back();
    //     }

    //     self.trail.push_front(SmokePoint {
    //         pos: current_orientation.translation.vector.xyz(),
    //         color: glm::vec4(1f32, 0f32, 0f32, 1f32),
    //     });
    // }
}

#[derive(
    Copy,
    Clone,
    Hash,
    Eq,
    PartialEq,
    strum_macros::EnumProperty,
    strum_macros::EnumIter,
    strum_macros::Display,
)]
#[repr(u8)]
pub enum ProjectileKind {
    #[strum(props(kind = "plasmabolt"))]
    Plasmabolt,
}

#[derive(Copy, Clone)]
pub struct ProjectileSpawnData {
    pub orientation: nalgebra::Isometry3<f32>,
    pub kind: ProjectileKind,
}

#[derive(Copy, Clone)]
pub struct MissileSpawnData {
    pub kind: MissileKind,
    pub initial_orientation: nalgebra::Isometry3<f32>,
    pub linear_vel: glm::Vec3,
    pub angular_vel: glm::Vec3,
}

#[derive(Copy, Clone)]
struct Projectile {
    kind: ProjectileKind,
    orientation: nalgebra::Isometry3<f32>,
    rigid_body: RigidBodyHandle,
    collider: ColliderHandle,
    life: f32,
    visible: bool,
}

struct ProjectileClassSheet {
    id: SubmeshId,
    mass: f32,
    aabb: AABB3,
    speed: f32,
    life: f32,
}

pub struct MissileSys {
    mesh_id: MeshId,
    live_missiles: Vec<Missile>,
    missile_classes: HashMap<MissileKind, MissileClassSheet>,
    projectiles: Vec<Projectile>,
    projectile_classes: HashMap<ProjectileKind, ProjectileClassSheet>,
}

impl MissileSys {
    const MAX_OBJECTS: u32 = 1024;

    pub fn new(init_ctx: &InitContext) -> Option<MissileSys> {
        let mesh_id: MeshId = "r73r27".into();
        let missile_mesh = init_ctx.rsys.get_mesh_info(mesh_id);

        use strum::IntoEnumIterator;

        let missile_classes = MissileKind::iter()
            .map(|msl_kind| {
                use strum::EnumProperty;
                let id: SubmeshId = msl_kind.get_str("kind").unwrap().into();
                let missile_node = missile_mesh.get_node(id);

                (
                    msl_kind,
                    MissileClassSheet {
                        id,
                        mass: 500f32,
                        aabb: missile_node.aabb,
                        booster_life: 25f32,
                        thrust: 550f32,
                    },
                )
            })
            .collect::<std::collections::HashMap<_, _>>();

        let projectile_classes = ProjectileKind::iter()
            .map(|proj_kind| {
                use strum::EnumProperty;
                let id: SubmeshId = proj_kind.get_str("kind").unwrap().into();
                let proj_node = missile_mesh.get_node(id);

                (
                    proj_kind,
                    ProjectileClassSheet {
                        id,
                        mass: 10f32,
                        aabb: proj_node.aabb,
                        speed: 2000f32,
                        life: 5f32,
                    },
                )
            })
            .collect::<std::collections::HashMap<_, _>>();

        Some(MissileSys {
            live_missiles: Vec::new(),
            projectiles: Vec::new(),
            missile_classes,
            projectile_classes,
            mesh_id,
        })
    }

    pub fn draw(&self, _draw_context: &DrawContext, draw_sys: &mut DrawingSys) {
        self.live_missiles
            .iter()
            .filter(|msl| !msl.out_of_vis_range)
            .for_each(|msl| {
                let msl_class_data = self.missile_classes.get(&msl.kind).unwrap();
                draw_sys.add_mesh(
                    self.mesh_id,
                    Some(msl_class_data.id),
                    None,
                    &msl.orientation.to_matrix(),
                    EffectType::BasicEmissive,
                );
            });

        self.projectiles.iter().filter(|p| p.visible).for_each(|p| {
            let p_class_data = self.projectile_classes.get(&p.kind).unwrap();
            draw_sys.add_mesh(
                self.mesh_id,
                Some(p_class_data.id),
                None,
                &p.orientation.to_matrix(),
                EffectType::BasicEmissive,
            );
        });
    }

    pub fn spawn_projectile(
        &mut self,
        spawn_data: &ProjectileSpawnData,
        physics_engine: &mut PhysicsEngine,
    ) {
        if self.projectiles.len() as u32 > Self::MAX_OBJECTS {
            log::error!("Can't spawn projectile, max object limit reached");
            return;
        }

        let p_class_data = self.projectile_classes.get(&spawn_data.kind).unwrap();
        let direction = (spawn_data.orientation * glm::Vec3::z_axis()).xyz();
        let velocity = direction * p_class_data.speed;

        let mut rigid_body = rapier3d::prelude::RigidBodyBuilder::dynamic()
            .position(spawn_data.orientation)
            .lock_rotations()
            .build();

        rigid_body.add_force(velocity, true);

        let rigid_body_handle = physics_engine.rigid_body_set.insert(rigid_body);
        let collider_extents = p_class_data.aabb.extents();

        let collider = rapier3d::prelude::ColliderBuilder::cuboid(
            collider_extents.x,
            collider_extents.y,
            collider_extents.z,
        )
        .active_events(rapier3d::prelude::ActiveEvents::COLLISION_EVENTS)
        .collision_groups(PhysicsObjectCollisionGroups::projectiles())
        .mass(p_class_data.mass)
        .sensor(true)
        .build();

        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_engine.rigid_body_set,
        );

        self.projectiles.push(Projectile {
            kind: spawn_data.kind,
            orientation: spawn_data.orientation,
            rigid_body: rigid_body_handle,
            collider: collider_handle,
            life: p_class_data.life,
            visible: true,
        });
    }

    pub fn spawn_missile(
        &mut self,
        spawn_data: &MissileSpawnData,
        physics_engine: &mut PhysicsEngine,
    ) {
        if self.live_missiles.len() as u32 >= Self::MAX_OBJECTS {
            log::error!("Cannot spawn missile, object limit reached");
            return;
        }

        let msl_class_sheet = self.missile_classes.get(&spawn_data.kind).expect(&format!(
            "Missing data sheet for missile class {}",
            spawn_data.kind
        ));

        use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyType};
        let body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
            .position(spawn_data.initial_orientation)
            .build();

        let body_handle = physics_engine.rigid_body_set.insert(body);

        let bbox_half_extents = msl_class_sheet.aabb.extents();
        let collider = ColliderBuilder::cuboid(
            bbox_half_extents.x,
            bbox_half_extents.y,
            bbox_half_extents.z,
        )
        .mass(msl_class_sheet.mass)
        .active_events(rapier3d::prelude::ActiveEvents::COLLISION_EVENTS)
        .collision_groups(PhysicsObjectCollisionGroups::missiles())
        .sensor(true)
        .build();

        let collider_handle = physics_engine.collider_set.insert_with_parent(
            collider,
            body_handle,
            &mut physics_engine.rigid_body_set,
        );

        let body = physics_engine.get_rigid_body_mut(body_handle);
        body.set_linvel(spawn_data.linear_vel, true);
        body.set_angvel(spawn_data.angular_vel, true);

        self.live_missiles.push(Missile {
            kind: spawn_data.kind,
            orientation: spawn_data.initial_orientation,
            booster_time: msl_class_sheet.booster_life,
            rigid_body: body_handle,
            collider: collider_handle,
            thrust: msl_class_sheet.thrust,
            out_of_vis_range: false,
        });
    }

    pub fn update(&mut self, context: &mut UpdateContext) {
        //
        // TODO: warhead arming only after a certain distance
        // TODO: actual missile logic
        self.live_missiles.retain_mut(|msl| {
            if msl.booster_time > 0f32 {
                msl.booster_time = (msl.booster_time - context.frame_time as f32).max(0f32);
            }

            if msl.booster_time > 0f32 {
                let msl_phys_body = context.physics_engine.get_rigid_body_mut(msl.rigid_body);
                msl.orientation = *msl_phys_body.position();
                //
                // apply force
                msl_phys_body.apply_impulse(msl.orientation * glm::Vec3::z() * msl.thrust, true);
                if !msl.out_of_vis_range {
                    let sqr_dist =
                        glm::distance2(&context.camera_pos, &msl.orientation.translation.vector);
                    const MAX_DRAW_DST: f32 = 1000f32 * 1000f32;
                    if sqr_dist > MAX_DRAW_DST {
                        msl.out_of_vis_range = true;
                    }
                }

                true
            } else {
                //
                // dead missile, remove it
                context.physics_engine.remove_rigid_body(msl.rigid_body);
                log::info!("Missile died @ {}", msl.orientation.translation.vector);
                false
            }
        });

        //
        // projectiles
        self.projectiles.retain_mut(|proj| {
            proj.life -= context.frame_time as f32;
            proj.orientation = *context
                .physics_engine
                .get_rigid_body(proj.rigid_body)
                .position();

            if proj.life > 0f32 {
                if proj.visible {
                    const MAX_DRAW_DST: f32 = 1000f32 * 1000f32;
                    proj.visible =
                        glm::distance2(&context.camera_pos, &proj.orientation.translation.vector)
                            < MAX_DRAW_DST;
                }

                true
            } else {
                context.physics_engine.remove_rigid_body(proj.rigid_body);
                false
            }
        });
    }

    pub fn despawn_projectile(&mut self, proj_body: RigidBodyHandle) {
        self.projectiles
            .iter()
            .position(|projectile| projectile.rigid_body == proj_body)
            .map(|proj_pos| {
                self.projectiles.swap_remove(proj_pos);
            });
    }
}
