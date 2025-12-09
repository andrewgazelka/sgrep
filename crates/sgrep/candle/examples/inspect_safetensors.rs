//! Inspect safetensors file to see tensor names.

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let path = "scripts/convert/models/model.safetensors";
    let tensors = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;

    let mut keys: Vec<_> = tensors.keys().collect();
    keys.sort();

    for key in &keys {
        let tensor = tensors.get(*key).unwrap();
        println!("{}: {:?}", key, tensor.dims());
    }

    println!("\nTotal tensors: {}", keys.len());

    Ok(())
}
