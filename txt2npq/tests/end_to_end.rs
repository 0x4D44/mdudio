use std::fs::File;
use std::io::{Read, Write};
use std::process::Command;
use tempfile::tempdir;

#[test]
fn test_end_to_end_round_trip() {
    let temp_dir = tempdir().unwrap();
    let input_txt_path = temp_dir.path().join("input.txt");
    let intermediate_npq_path = temp_dir.path().join("input.npq"); // Derived from input_txt_path
    let final_txt_path = temp_dir.path().join("input.txt"); // Derived from intermediate_npq_path

    // 1. Create a known input text file
    let input_text = r###"Magic: NPQ1
Version: 1
Num Codebooks (K): 8
Token Rate (fps): 86.13
Original Bitrate (kbps): 320.0
Sequence Length (T): 2
Vocab Sizes: [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
Dtype Code: 1

Payload:
T=0: [1, 2, 3, 4, 5, 6, 7, 8]
T=1: [9, 10, 11, 12, 13, 14, 15, 16]"###;
    File::create(&input_txt_path)
        .unwrap()
        .write_all(input_text.as_bytes())
        .unwrap();

    // 2. Run txt2npq
    let txt2npq_output = Command::new("c:\\apps\\txt2npq.exe")
        .arg(&input_txt_path) // Only input file as argument
        .output()
        .expect("Failed to execute txt2npq. Make sure it is in your PATH.");

    if !txt2npq_output.status.success() {
        println!("txt2npq stdout: {}", String::from_utf8_lossy(&txt2npq_output.stdout));
        println!("txt2npq stderr: {}", String::from_utf8_lossy(&txt2npq_output.stderr));
    }
    assert!(txt2npq_output.status.success());

    // 3. Run npq2txt on the generated .npq file
    let npq2txt_output = Command::new("c:\\apps\\npq2txt.exe")
        .arg(&intermediate_npq_path) // Only input file as argument
        .output()
        .expect("Failed to execute npq2txt. Make sure it is in your PATH.");

    if !npq2txt_output.status.success() {
        println!("npq2txt stdout: {}", String::from_utf8_lossy(&npq2txt_output.stdout));
        println!("npq2txt stderr: {}", String::from_utf8_lossy(&npq2txt_output.stderr));
    }
    assert!(npq2txt_output.status.success());

    // 4. Compare the output of npq2txt with the original input text
    let mut final_output_file = File::open(&final_txt_path).unwrap();
    let mut final_output_text = String::new();
    final_output_file.read_to_string(&mut final_output_text).unwrap();

    // Normalize line endings for comparison
    let normalized_final_output_text = final_output_text.replace("\r\n", "\n");

    // npq2txt adds "Header:" and "Payload:" and "Payload (first X rows):"
    // We need to parse the generated text to extract just the header and payload for comparison.
    // This is a simplified comparison, a more robust solution would parse both.
    // For now, let's just check if the key parts are present.

    assert!(normalized_final_output_text.contains("Magic: NPQ1"));
    assert!(normalized_final_output_text.contains("Version: 1"));
    assert!(normalized_final_output_text.contains("Num Codebooks (K): 8"));
    assert!(normalized_final_output_text.contains("Token Rate (fps): 86.13"));
    assert!(normalized_final_output_text.contains("Original Bitrate (kbps): 320")); // npq2txt might format 320.0 as 320
    assert!(normalized_final_output_text.contains("Sequence Length (T): 2"));
    assert!(normalized_final_output_text.contains("Vocab Sizes: [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]"));
    assert!(normalized_final_output_text.contains("Dtype Code: 1"));
    assert!(normalized_final_output_text.contains("T=0: [1, 2, 3, 4, 5, 6, 7, 8]"));
    assert!(normalized_final_output_text.contains("T=1: [9, 10, 11, 12, 13, 14, 15, 16]"));
}
