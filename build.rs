use std::env;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn add_extension(path: &mut std::path::PathBuf, extension: impl AsRef<std::path::Path>) {
    match path.extension() {
        Some(ext) => {
            let mut ext = ext.to_os_string();
            ext.push(".");
            ext.push(extension.as_ref());
            path.set_extension(ext)
        }
        None => path.set_extension(extension.as_ref()),
    };
}

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-changed=data/shaders");

    let shader_source_files = std::fs::read_dir("src/shaders")?
        .filter_map(|dir_entry| dir_entry.ok().map(|de| de.path()))
        .filter(|f| {
            f.is_file()
                && f.extension()
                    .map(|ext| {
                        ["vert", "geom", "frag"]
                            .iter()
                            .any(|&shader_ext| shader_ext == ext.to_string_lossy().as_ref())
                    })
                    .unwrap_or_default()
        })
        .collect::<Vec<_>>();

    let out_dir =
        Path::new(&env::var("OUT_DIR").expect("Failed to read OUT_DIR env var")).join("shaders");
    let bytecode_output_dir = Path::new("data/shaders");

    let _ = std::fs::create_dir(out_dir.clone());
    let _ = std::fs::create_dir_all(bytecode_output_dir);

    // println!("cargo:warning=Shader files {:?}", shader_source_files);

    let compiled_shaders = shader_source_files
        .iter()
        .filter_map(|shader_file| {
            let mut output_file = PathBuf::new();
            output_file.push(shader_file.file_name()?);
            add_extension(&mut output_file, "spv");

            let out_file_name = out_dir.join(output_file);

            Command::new("glslc")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .arg("-g")
                .arg("-O0")
                .arg("--target-env=vulkan1.3")
                .arg("--target-spv=spv1.6")
                .arg("-std=460core")
                .arg("-o")
                .arg(out_file_name.as_os_str().to_str()?)
                .arg(shader_file.as_os_str().to_str()?)
                .spawn()
                .ok()
                .map(|compile_process| (compile_process, out_file_name))
        })
        .filter_map(|(compile_process, bytecode_file_name)| {
            compile_process
                .wait_with_output()
                .ok()
                .and_then(|glslc_output| {
                    glslc_output.status.code().map(|glslc_exit_code| {
                        if glslc_exit_code == 0 {
                            Some(())
                        } else {
                            println!(
                                "cargo:warning={}",
                                unsafe { CString::from_vec_unchecked(glslc_output.stderr) }
                                    .to_str()
                                    .unwrap()
                            );
                            None
                        }
                    })
                })
                .flatten()
                .and_then(|_| {
                    let dest_bytecode_filename = bytecode_file_name.file_name()?;
                    let bytecode_copy_path = Path::new("data/shaders")
                        // get_output_path()
                        // .join("shaders")
                        .join(dest_bytecode_filename);

                    std::fs::copy(bytecode_file_name.clone(), bytecode_copy_path)
                        .map_err(|e| println!("cargo:warning=Copy bytecode failed: {}", e))
                        .ok()
                })
        })
        .count();

    // println!("cargo:warning=Compiled shaders: {}", compiled_shaders);

    if compiled_shaders == shader_source_files.len() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Shader bytecode compilation failed",
        ))
    }
}
