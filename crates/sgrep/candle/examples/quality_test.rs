//! Quality test for semantic similarity

use eyre::WrapErr as _;

const MODEL_DIR: &str = "scripts/convert/models";

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let model_path = format!("{MODEL_DIR}/model.safetensors");
    let tokenizer_path = format!("{MODEL_DIR}/tokenizer.json");
    let config_path = format!("{MODEL_DIR}/config.json");

    let mut encoder =
        sgrep_candle::ColBertEncoder::load(&model_path, &tokenizer_path, &config_path)?;

    println!("=== Semantic Quality Tests ===\n");

    // Test 1: Code concept matching
    println!("Test 1: Code concept matching");
    let query = encoder.encode_query("error handling")?;
    let docs = [
        ("try { } catch (e) { handle(e); }", "error handling code"),
        ("Result<T, E> with ? operator", "Rust error handling"),
        ("if err != nil { return err }", "Go error handling"),
        ("hello world greeting", "unrelated"),
        ("for i in range(10): print(i)", "loop code"),
    ];

    let mut scores: Vec<_> = docs
        .iter()
        .map(|(code, desc)| {
            let emb = encoder.encode_document(code).unwrap();
            let score = sgrep_embed::maxsim(&query, &emb).unwrap();
            (score, *desc, *code)
        })
        .collect();
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("  Query: \"error handling\"");
    for (score, desc, code) in &scores {
        let marker = if desc.contains("error") { "✓" } else { " " };
        println!("  {marker} {score:.3} [{desc}] {code}");
    }

    // Test 2: Function search
    println!("\nTest 2: Function search");
    let query = encoder.encode_query("main function entry point")?;
    let docs = [
        ("fn main() { app.run(); }", "Rust main"),
        ("def main(): pass", "Python main"),
        ("int main(int argc, char** argv)", "C main"),
        ("class UserService {}", "unrelated class"),
        ("const x = 42;", "variable declaration"),
    ];

    let mut scores: Vec<_> = docs
        .iter()
        .map(|(code, desc)| {
            let emb = encoder.encode_document(code).unwrap();
            let score = sgrep_embed::maxsim(&query, &emb).unwrap();
            (score, *desc, *code)
        })
        .collect();
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("  Query: \"main function entry point\"");
    for (score, desc, code) in &scores {
        let marker = if desc.contains("main") { "✓" } else { " " };
        println!("  {marker} {score:.3} [{desc}] {code}");
    }

    // Test 3: Database operations
    println!("\nTest 3: Database operations");
    let query = encoder.encode_query("database query select")?;
    let docs = [
        ("SELECT * FROM users WHERE id = ?", "SQL select"),
        ("db.find({ name: 'John' })", "MongoDB query"),
        ("cursor.execute('SELECT...')", "Python DB"),
        ("println!(\"hello\")", "print statement"),
        ("let x = vec![1,2,3];", "vector creation"),
    ];

    let mut scores: Vec<_> = docs
        .iter()
        .map(|(code, desc)| {
            let emb = encoder.encode_document(code).unwrap();
            let score = sgrep_embed::maxsim(&query, &emb).unwrap();
            (score, *desc, *code)
        })
        .collect();
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("  Query: \"database query select\"");
    for (score, desc, code) in &scores {
        let marker = if desc.contains("SQL") || desc.contains("DB") || desc.contains("Mongo") {
            "✓"
        } else {
            " "
        };
        println!("  {marker} {score:.3} [{desc}] {code}");
    }

    // Test 4: HTTP/API
    println!("\nTest 4: HTTP/API");
    let query = encoder.encode_query("http request api call")?;
    let docs = [
        ("fetch('/api/users').then(r => r.json())", "JS fetch"),
        ("requests.get('https://api.example.com')", "Python requests"),
        ("HttpClient::new().get(url).send()?", "Rust HTTP"),
        ("for item in items { process(item); }", "loop"),
        ("struct Point { x: i32, y: i32 }", "struct definition"),
    ];

    let mut scores: Vec<_> = docs
        .iter()
        .map(|(code, desc)| {
            let emb = encoder.encode_document(code).unwrap();
            let score = sgrep_embed::maxsim(&query, &emb).unwrap();
            (score, *desc, *code)
        })
        .collect();
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("  Query: \"http request api call\"");
    for (score, desc, code) in &scores {
        let marker = if desc.contains("fetch") || desc.contains("HTTP") || desc.contains("requests")
        {
            "✓"
        } else {
            " "
        };
        println!("  {marker} {score:.3} [{desc}] {code}");
    }

    // Test 5: Async/concurrent
    println!("\nTest 5: Async/concurrent code");
    let query = encoder.encode_query("async await concurrent")?;
    let docs = [
        ("async fn fetch() -> Result<()> { Ok(()) }", "Rust async"),
        ("await Promise.all(tasks)", "JS Promise.all"),
        ("go func() { ch <- result }()", "Go goroutine"),
        ("let x = 1 + 2;", "arithmetic"),
        ("if condition { do_thing(); }", "conditional"),
    ];

    let mut scores: Vec<_> = docs
        .iter()
        .map(|(code, desc)| {
            let emb = encoder.encode_document(code).unwrap();
            let score = sgrep_embed::maxsim(&query, &emb).unwrap();
            (score, *desc, *code)
        })
        .collect();
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("  Query: \"async await concurrent\"");
    for (score, desc, code) in &scores {
        let marker =
            if desc.contains("async") || desc.contains("Promise") || desc.contains("goroutine") {
                "✓"
            } else {
                " "
            };
        println!("  {marker} {score:.3} [{desc}] {code}");
    }

    println!("\n✓ = expected relevant result");

    Ok(())
}
