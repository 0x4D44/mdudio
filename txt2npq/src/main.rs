use clap::Parser;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// The path to the input text file
    input_file: PathBuf,
}

fn from_crockford_base32(s: &str) -> Result<u32, &'static str> {
    if s.is_empty() {
        return Err("Empty string");
    }
    
    let mut result = 0u32;
    for c in s.chars() {
        let digit = match c {
            '0' => 0, '1' => 1, '2' => 2, '3' => 3, '4' => 4,
            '5' => 5, '6' => 6, '7' => 7, '8' => 8, '9' => 9,
            'A' | 'a' => 10, 'B' | 'b' => 11, 'C' | 'c' => 12, 'D' | 'd' => 13,
            'E' | 'e' => 14, 'F' | 'f' => 15, 'G' | 'g' => 16, 'H' | 'h' => 17,
            'J' | 'j' => 18, 'K' | 'k' => 19, 'M' | 'm' => 20, 'N' | 'n' => 21,
            'P' | 'p' => 22, 'Q' | 'q' => 23, 'R' | 'r' => 24, 'S' | 's' => 25,
            'T' | 't' => 26, 'V' | 'v' => 27, 'W' | 'w' => 28, 'X' | 'x' => 29,
            'Y' | 'y' => 30, 'Z' | 'z' => 31,
            _ => return Err("Invalid character"),
        };
        result = result * 32 + digit;
    }
    Ok(result)
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let input_path = &cli.input_file;
    let output_path = input_path.with_extension("npq");

    let input_file = File::open(input_path)?;
    let reader = BufReader::new(input_file);

    let mut magic = String::new();
    let mut version = 0;
    let mut num_codebooks = 0;
    let mut token_rate = 0.0;
    let _original_bitrate = 0.0;
    let mut sequence_length = 0;
    let mut vocab_sizes = Vec::new();
    let mut dtype_code = 0;
    let mut payload: Vec<Vec<u32>> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('#') {
            // Parse header lines like "# magic: NPQ1"
            if let Some(colon_pos) = line.find(':') {
                let key = line[1..colon_pos].trim(); // Remove '#' and get key
                let value = line[colon_pos+1..].trim();
                
                match key {
                    "magic" => magic = value.to_string(),
                    "version" => version = value.parse().unwrap_or(0),
                    "num_codebooks" => num_codebooks = value.parse().unwrap_or(0),
                    "token_rate_hz" => token_rate = value.parse().unwrap_or(0.0),
                    "seq_len_frames" => sequence_length = value.parse().unwrap_or(0),
                    "vocab_sizes" => {
                        vocab_sizes = value.split(',').map(|s| s.trim().parse().unwrap_or(0)).collect();
                    }
                    "dtype" => {
                        dtype_code = match value {
                            "uint8" => 0,
                            "uint16" => 1,
                            "uint32" => 2,
                            _ => 0
                        };
                    }
                    _ => {}
                }
            }
        } else {
            // Parse payload lines like "a0F b03 cQ1 d0F eMX f2A g9T h4K i1V"
            let token_parts: Vec<&str> = line.split_whitespace().collect();
            let mut tokens: Vec<u32> = Vec::new();
            
            for token_part in token_parts {
                if token_part.len() >= 2 {
                    // Extract token value (everything after first character)
                    let value_str = &token_part[1..];
                    if let Ok(token) = from_crockford_base32(value_str) {
                        tokens.push(token as u32);
                    }
                }
            }
            
            if !tokens.is_empty() {
                payload.push(tokens);
            }
        }
    }

    let mut output_file = File::create(output_path)?;
    output_file.write_all(magic.as_bytes())?;
    output_file.write_u16::<LittleEndian>(version)?;
    output_file.write_u16::<LittleEndian>(num_codebooks)?;
    output_file.write_f32::<LittleEndian>(token_rate)?;
    output_file.write_f32::<LittleEndian>(_original_bitrate)?;
    output_file.write_u32::<LittleEndian>(sequence_length)?;
    for &size in &vocab_sizes {
        output_file.write_u32::<LittleEndian>(size)?;
    }
    output_file.write_u8(dtype_code)?;

    for tokens in &payload {
        for &token in tokens {
            match dtype_code {
                0 => output_file.write_u8(token as u8)?,
                1 => output_file.write_u16::<LittleEndian>(token as u16)?,
                2 => output_file.write_u32::<LittleEndian>(token)?,
                _ => return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid dtype_code",
                )),
            }
        }
    }

    Ok(())
}