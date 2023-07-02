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

#[derive(Serialize, Deserialize)]
pub struct PlayerShipConfig {
    pub crosshair_normal: String,
    pub crosshair_hit: String,
    pub target_outline: String,
    pub target_centermass: String,
    pub crosshair_size: f32,
    pub crosshair_color: u32,
    pub target_color: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SceneDescription {
    pub skyboxes: Vec<SkyboxDescription>,
    pub geometry: Vec<GeometryDescription>,
}

pub struct AppConfig {
    pub engine: EngineConfig,
    pub scene: SceneDescription,
    pub player: PlayerShipConfig,
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
            player: ron::de::from_reader(
                File::open("config/player.config.ron").expect("Failed to open player config file"),
            )
            .expect("Can't parse player config file!"),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EngineConfig {
    pub full_screen: bool,
    pub root_path: PathBuf,
    pub textures: PathBuf,
    pub models: PathBuf,
    pub shaders: PathBuf,
    pub fonts: PathBuf,
}

impl EngineConfig {
    pub fn shader_path<P: AsRef<std::path::Path>>(&self, shader_file: P) -> PathBuf {
        self.shaders.clone().join(shader_file)
    }

    pub fn texture_path<P: AsRef<std::path::Path>>(&self, texture_file: P) -> PathBuf {
        self.textures.clone().join(texture_file)
    }

    pub fn fonts_path<P: AsRef<std::path::Path>>(&self, font_file: P) -> PathBuf {
        self.fonts.clone().join(font_file)
    }
}
