use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::plan::{builtin_profiles, PROFILES_SCHEMA};
use serde_json::json;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "List the built-in Weight-Space Programming research profiles with their subspace definitions, default target activation shifts, and layer-selection rules.",
    after_help = "Example:\n  secant profiles\n  secant profiles --json"
)]
pub struct Args {
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let profiles = builtin_profiles();
    if a.json {
        let v = json!({
            "schema": PROFILES_SCHEMA,
            "count": profiles.len(),
            "profiles": profiles,
        });
        return match emit_json(&v) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    println!("Weight-Space Programming profiles ({}):\n", profiles.len());
    for p in &profiles {
        println!("{}\n  {}", p.name, p.description);
        let names: Vec<&str> = p.subspaces.iter().map(|s| s.name.as_str()).collect();
        println!("  subspaces:     {}", names.join(", "));
        if p.default_target_shift != 0.0 {
            println!("  default shift: {:+.1} (target activation shift)", p.default_target_shift);
        }
        if p.default_gamma != 0.0 {
            println!("  default gamma: {:.2} (relative Frobenius norm)", p.default_gamma);
        }
        println!("  layer choice:  {} ({})", p.layer_selection.strategy, p.layer_selection.tool);
        let slots: Vec<String> = p
            .slots
            .iter()
            .map(|s| format!("{}({}@L{})", s.id, s.role, s.layer))
            .collect();
        println!("  slots:         {}\n", slots.join(", "));
    }
    0
}
