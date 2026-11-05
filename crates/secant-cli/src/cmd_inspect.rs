use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::container::inspect_model;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Model checkpoint (.gguf or .safetensors)
    #[arg(long)]
    model: Option<String>,
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
    println!("{}: {}, {} tensors, {} bytes, payload at +{}", rep.path, rep.format, rep.tensor_count, rep.file_size, rep.data_offset);
    if !rep.architecture.is_empty() {
        print!("architecture {}, {} blocks", rep.architecture, rep.block_count);
        if rep.hidden_size != 0 {
            print!(", hidden {}", rep.hidden_size);
        }
        println!();
    }
    for (dt, n) in &rep.dtype_counts {
        println!("  {:<4} {} tensor(s)", dt, n);
    }
    if let Some(s) = &rep.qwen35_gdn_ssm_out {
        if s.detected {
            println!("qwen3.5 GDN ssm_out: {} layer(s), logical [{} x {}] {}", s.layers.len(), s.out_width, s.in_width, s.dtype);
            println!("  {}", s.description);
        }
    }
    for n in &rep.notes {
        println!("note: {}", n);
    }
    0
}
