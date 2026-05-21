use burn::backend::{Autodiff, Cuda};

mod data;
mod infer;
mod model;
mod tokenizer;
mod training;

type MyBackend = Autodiff<Cuda>;

#[derive(Debug)]
enum Mode {
    Train {
        data_path: Option<String>,
    },
    Infer {
        prompt: String,
        config: infer::GenerateConfig,
    },
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match parse_args(&args) {
        Mode::Infer { prompt, config } => {
            eprintln!("🔄 Inference mode");
            eprintln!("Prompt: \"{}\"", prompt);
            eprintln!(
                "Steps: {} | Temp: {:.2} | Top-k: {} | Top-p: {:.2}",
                config.max_steps, config.temperature, config.top_k, config.top_p
            );

            let device = Default::default();
            let output = infer::generate::<MyBackend>(&device, &prompt, config);

            println!("\n=== Generated ===\n{output}");
        }

        Mode::Train { data_path } => {
            let data_path = data_path.unwrap_or_else(|| "data/train.txt".to_string());
            eprintln!("🏋️ Training mode on: {}", data_path);

            let device = Default::default();
            // Исправление здесь: .as_deref() -> Option<&str>
            training::run::<MyBackend>(device, Some(&data_path));
        }
    }
}

fn parse_args(args: &[String]) -> Mode {
    // Inference mode
    if args.iter().any(|a| a == "--infer" || a == "infer") {
        let prompt = get_prompt(args);

        let config = infer::GenerateConfig {
            max_steps: get_usize(args, "--steps", 100),
            temperature: get_f32(args, "--temp", 0.8),
            top_k: get_usize(args, "--topk", 40),
            top_p: get_f32(args, "--topp", 0.9),
            max_len: get_usize(args, "--maxlen", 128),
        };

        return Mode::Infer { prompt, config };
    }

    // Training mode (default)
    let data_path = get_string(args, "--data");
    Mode::Train { data_path }
}

// ==================== Helpers ====================

fn get_prompt(args: &[String]) -> String {
    if let Some(pos) = args.iter().position(|a| a == "--prompt") {
        if let Some(text) = args.get(pos + 1) {
            return text.clone();
        }
    }
    args.get(2).cloned().unwrap_or_else(|| "Привет".to_string())
}

fn get_string(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn get_usize(args: &[String], flag: &str, default: usize) -> usize {
    get_string(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn get_f32(args: &[String], flag: &str, default: f32) -> f32 {
    get_string(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
