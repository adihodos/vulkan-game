use std::{fs::File, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SkyboxDescription {
    pub tag: String,
    pub path: std::path::PathBuf,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GeometryDescription {
    pub tag: String,
    pub path: std::path::PathBuf,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SceneDescription {
    pub skyboxes: Vec<SkyboxDescription>,
    pub geometry: Vec<GeometryDescription>,
}

pub struct AppConfig {
    pub engine: EngineConfig,
    pub scene: SceneDescription,
}

impl AppConfig {
    pub fn load() -> AppConfig {
        AppConfig {
            engine: ron::de::from_reader(
                File::open("config/engine.config.ron").expect("Can't open engine config file!"),
            )
            .expect("Failed to read engine configuration"),
            scene: ron::de::from_reader(
                File::open("config/scene.config.ron").expect("Can't open scene config file!"),
            )
            .expect("Failed to read scene description"),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EngineConfig {
    pub root_path: PathBuf,
    pub textures: PathBuf,
    pub models: PathBuf,
    pub shaders: PathBuf,
}

impl EngineConfig {
    pub fn shader_path<P: AsRef<std::path::Path>>(&self, shader_file: P) -> PathBuf {
        self.shaders.clone().join(shader_file)
    }
}

// fn write_config() {
//     use ron::ser::{to_writer_pretty, PrettyConfig};
//
//     let engine_cfg = EngineConfig {
//         root_path: "data".into(),
//         textures: "data/textures".into(),
//         models: "data/models".into(),
//         shaders: "data/shaders".into(),
//     };
//
//     let cfg_opts = PrettyConfig::new()
//         .depth_limit(8)
//         .separate_tuple_members(true);
//
//     to_writer_pretty(
//         File::create("config/engine.cfg.ron").expect("cykaaaaa"),
//         &engine_cfg,
//         cfg_opts.clone(),
//     )
//     .expect("oh noes ...");
//
//     let my_scene = SceneDescription {
//         skyboxes: vec![SkyboxDescription {
//             tag: "starfield1".into(),
//             path: "skybox-ibl".into(),
//         }],
//     };
//
//     to_writer_pretty(
//         File::create("config/scene.cfg.ron").expect("kurwa jebane!"),
//         &my_scene,
//         cfg_opts,
//     )
//     .expect("Dublu plm ,,,");
// }
