use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    capnpc::CompilerCommand::new()
        .file("schema/inference.capnp")
        .output_path(&out_dir)
        .run()
        .expect("compiling capnp schema");
}
