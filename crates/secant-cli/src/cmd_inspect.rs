use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::container::inspect_model;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Inspect a GGUF or SafeTensors checkpoint without changing it. Report architecture, dtype counts, file size and payload offsets, including detected Qwen3.5 interleaving.",
    after_help = "Example:\n  secant inspect --model model.safetensors --json"
)]
pub struct Args {
    /// Model checkpoint (.gguf or .safetensors)
    #[arg(long)]
    model: Option<String>,
    /// Emit the stable machine-readable JSON report
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let Some(model) = &a.model else { return usage("--model is required") };
    let rep = match inspect_model(model) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if a.json {
        return match emit_json(&rep) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    section("Checkpoint layout");
    field("Model", &rep.path);
    field("Format", &rep.format);
    field("Tensors", rep.tensor_count);
    field("File size", bytes(rep.file_size as u64));
    field("Payload offset", bytes(rep.data_offset as u64));
    field("Architecture", if rep.architecture.is_empty() { "UNKNOWN" } else { &rep.architecture });
    if rep.block_count != 0 {
        field("Blocks", rep.block_count);
    }
    if rep.hidden_size != 0 {
        field("Hidden size", rep.hidden_size);
    }
    if rep.context_length != 0 {
        field("Context length (header)", rep.context_length);
    }
    section("Tensor dtypes");
    table(&["DTYPE", "TENSORS"], &rep.dtype_counts.iter().map(|(dt, n)| vec![dt.clone(), n.to_string()]).collect::<Vec<_>>());
    if let Some(s) = &rep.qwen35_gdn_ssm_out {
        if s.detected {
            section("Qwen3.5 GDN ssm_out");
            field("Layers", s.layers.len());
            field("Logical shape", format!("[{} x {}]", s.out_width, s.in_width));
            field("Dtype", &s.dtype);
            field("Interleave", &s.description);
        }
    }
    for n in &rep.notes {
        println!("note: {}", n);
    }
    0
}
