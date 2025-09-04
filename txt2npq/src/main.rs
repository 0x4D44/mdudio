use clap::Parser;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// The path to the input text file
    input_file: PathBuf,
}

fn from_crockford_base32(s: &str) -> Result<u32, String> {
    if s.is_empty() {
        return Err("empty token".to_string());
    }

    let mut result = 0u32;
    for (idx, c) in s.chars().enumerate() {
        let digit = match c {
            '0' => 0, '1' => 1, '2' => 2, '3' => 3, '4' => 4,
            '5' => 5, '6' => 6, '7' => 7, '8' => 8, '9' => 9,
            'A' | 'a' => 10, 'B' | 'b' => 11, 'C' | 'c' => 12, 'D' | 'd' => 13,
            'E' | 'e' => 14, 'F' | 'f' => 15, 'G' | 'g' => 16, 'H' | 'h' => 17,
            'J' | 'j' => 18, 'K' | 'k' => 19, 'M' | 'm' => 20, 'N' | 'n' => 21,
            'P' | 'p' => 22, 'Q' | 'q' => 23, 'R' | 'r' => 24, 'S' | 's' => 25,
            'T' | 't' => 26, 'V' | 'v' => 27, 'W' | 'w' => 28, 'X' | 'x' => 29,
            'Y' | 'y' => 30, 'Z' | 'z' => 31,
            _ => return Err(format!("invalid base32 char '{}' at pos {}", c, idx)),
        };
        result = result
            .checked_mul(32)
            .and_then(|r| r.checked_add(digit))
            .ok_or_else(|| "token overflow".to_string())?;
    }
    Ok(result)
}

fn parse_header_line(line: &str, line_no: usize,
    magic: &mut Option<String>,
    version: &mut Option<u16>,
    num_codebooks: &mut Option<u16>,
    token_rate: &mut Option<f32>,
    sequence_length: &mut Option<u32>,
    vocab_sizes: &mut Option<Vec<u32>>,
    dtype_code: &mut Option<u8>,
) -> Result<(), io::Error> {
    if let Some(colon_pos) = line.find(':') {
        let key = line[1..colon_pos].trim();
        let value = line[colon_pos + 1..].trim();

        match key {
            "magic" => {
                *magic = Some(value.to_string());
            }
            "version" => {
                let v: u16 = value.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("line {}: invalid version '{}'" , line_no, value)))?;
                *version = Some(v);
            }
            "num_codebooks" => {
                let k: u16 = value.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("line {}: invalid num_codebooks '{}'", line_no, value)))?;
                *num_codebooks = Some(k);
            }
            "token_rate_hz" => {
                let tr: f32 = value.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("line {}: invalid token_rate_hz '{}'", line_no, value)))?;
                *token_rate = Some(tr);
            }
            "seq_len_frames" => {
                let t: u32 = value.parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("line {}: invalid seq_len_frames '{}'", line_no, value)))?;
                *sequence_length = Some(t);
            }
            "vocab_sizes" => {
                let list: Result<Vec<u32>, _> = value
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect();
                let list = list.map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("line {}: invalid vocab_sizes '{}'", line_no, value)))?;
                *vocab_sizes = Some(list);
            }
            "dtype" => {
                let code = match value {
                    "uint8" => 0u8,
                    "uint16" => 1u8,
                    "uint32" => 2u8,
                    other => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("line {}: unsupported dtype '{}'", line_no, other))),
                };
                *dtype_code = Some(code);
            }
            // Allow and ignore metadata lines we don't use
            "npq2txt v0.1" | "src_file" => {}
            _ => {
                eprintln!("warning: line {}: unknown header key '{}'", line_no, key);
            }
        }
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let input_path = &cli.input_file;
    let output_path = input_path.with_extension("npq");

    eprintln!("txt2npq: reading '{}', writing '{}'", input_path.display(), output_path.display());

    let input_file = File::open(input_path)?;
    let reader = BufReader::new(input_file);

    // Header fields (optional until validated)
    let mut magic: Option<String> = None;
    let mut version: Option<u16> = None;
    let mut num_codebooks: Option<u16> = None;
    let mut token_rate: Option<f32> = None;
    let original_bitrate: f32 = 0.0; // Not present in text format
    let mut sequence_length: Option<u32> = None;
    let mut vocab_sizes: Option<Vec<u32>> = None;
    let mut dtype_code: Option<u8> = None;

    let mut payload: Vec<Vec<u32>> = Vec::new();
    let mut seen_payload = false;
    let mut k_val: Option<usize> = None;
    let mut dtype_val: Option<u8> = None;

    for (line_no, line) in reader.lines().enumerate() {
        let line_no = line_no + 1; // make 1-based for messages
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() { continue; }

        if trimmed.starts_with('#') && !seen_payload {
            // Only parse header until we hit first payload row
            parse_header_line(trimmed, line_no, &mut magic, &mut version, &mut num_codebooks, &mut token_rate, &mut sequence_length, &mut vocab_sizes, &mut dtype_code)?;
            continue;
        }

        // First payload row encountered: validate header now
        if !seen_payload {
            seen_payload = true;

            // Borrow-only validation; don't consume Options here
            let mg = magic.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: magic"))?;
            if mg.as_bytes() != b"NPQ1" {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid magic '{}', expected NPQ1", mg)));
            }
            let ver = version.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: version"))?;
            let k = num_codebooks.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: num_codebooks"))?;
            let tr = token_rate.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: token_rate_hz"))?;
            let t = sequence_length.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: seq_len_frames"))?;
            let vs = vocab_sizes.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: vocab_sizes"))?;
            let dt = dtype_code.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: dtype"))?;

            if vs.len() != k as usize {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                    "vocab_sizes length {} does not match num_codebooks {}",
                    vs.len(), k
                )));
            }

            if !(1..=2).contains(&(dt as i32)) && dt != 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid dtype_code {}", dt)));
            }

            eprintln!(
                "header OK: magic=NPQ1, version={}, K={}, rate={:.4}Hz, T={}, dtype={}, vocab_sizes={:?}",
                ver,
                k,
                tr,
                t,
                match dt { 0 => "u8", 1 => "u16", 2 => "u32", _ => "?" },
                vs
            );

            // Cache some frequently used values for row parsing
            k_val = Some(k as usize);
            dtype_val = Some(dt);
        }

        // Parse one payload line
        let token_parts: Vec<&str> = trimmed.split_whitespace().collect();

        // We need num_codebooks for validation
        let k = k_val.unwrap_or(num_codebooks.unwrap_or(0) as usize);
        if token_parts.len() != k {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                "line {}: expected {} tokens, found {} (line='{}')",
                line_no, k, token_parts.len(), trimmed
            )));
        }

        let mut tokens: Vec<u32> = Vec::with_capacity(k);
        for (i, token_part) in token_parts.iter().enumerate() {
            let tp = token_part.trim();
            if tp.len() < 2 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                    "line {}: token '{}' too short",
                    line_no, tp
                )));
            }

            let expected_tag = (b'a' + i as u8) as char;
            let actual_tag = tp.chars().next().unwrap();
            if actual_tag != expected_tag {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                    "line {}: token '{}' has tag '{}', expected '{}'",
                    line_no, tp, actual_tag, expected_tag
                )));
            }

            let value_str = &tp[1..];
            let val = from_crockford_base32(value_str).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!(
                "line {}: token '{}' decode error: {}",
                line_no, tp, e
            )))?;

            // Range checks
            let vs = vocab_sizes.as_ref().unwrap();
            if (val as usize) >= vs[i] as usize {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                    "line {}: token '{}' value {} out of range (vocab_size {})",
                    line_no, tp, val, vs[i]
                )));
            }
            match dtype_val.unwrap_or(dtype_code.unwrap()) {
                0 if val > u8::MAX as u32 => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("line {}: token '{}' exceeds u8", line_no, tp))),
                1 if val > u16::MAX as u32 => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("line {}: token '{}' exceeds u16", line_no, tp))),
                2 => {}
                _ => {}
            }

            tokens.push(val);
        }

        payload.push(tokens);

        // Lightweight progress every ~5k frames
        if payload.len() % 5000 == 0 {
            let t = sequence_length.unwrap_or(0);
            if t > 0 {
                eprintln!("progress: {}/{} frames ({:.1}%)", payload.len(), t as usize, (payload.len() as f64) * 100.0 / (t as f64));
            } else {
                eprintln!("progress: {} frames", payload.len());
            }
        }

        // If file provides more rows than declared, warn but keep going; we'll correct header later.
        if let Some(t) = sequence_length {
            if (payload.len() as u32) > t {
                eprintln!(
                    "warning: line {}: payload row {} exceeds declared seq_len_frames {}",
                    line_no,
                    payload.len() - 1,
                    t
                );
            }
        }
    }

    // Final header validations now that we've read file
    let magic = magic.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: magic"))?;
    if magic.as_bytes() != b"NPQ1" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid magic '{}', expected NPQ1", magic)));
    }
    let version = version.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: version"))?;
    let num_codebooks = num_codebooks.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: num_codebooks"))?;
    let token_rate = token_rate.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: token_rate_hz"))?;
    let mut sequence_length = sequence_length.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: seq_len_frames"))?;
    let vocab_sizes = vocab_sizes.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: vocab_sizes"))?;
    let dtype_code = dtype_code.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing header: dtype"))?;

    if vocab_sizes.len() != num_codebooks as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
            "vocab_sizes length {} does not match num_codebooks {}",
            vocab_sizes.len(), num_codebooks
        )));
    }

    let actual_len = payload.len() as u32;
    if actual_len != sequence_length {
        eprintln!(
            "Payload {} doesn't match seq_len_frames {} - correcting seq_len_frames header",
            actual_len,
            sequence_length
        );
        sequence_length = actual_len;
    }

    // Write output
    let mut output_file = File::create(output_path)?;
    output_file.write_all(b"NPQ1")?; // magic must be exactly 4 bytes
    output_file.write_u16::<LittleEndian>(version)?;
    output_file.write_u16::<LittleEndian>(num_codebooks)?;
    output_file.write_f32::<LittleEndian>(token_rate)?;
    output_file.write_f32::<LittleEndian>(original_bitrate)?;
    output_file.write_u32::<LittleEndian>(sequence_length)?;
    for &size in &vocab_sizes {
        output_file.write_u32::<LittleEndian>(size)?;
    }
    output_file.write_u8(dtype_code)?;

    for (row_idx, tokens) in payload.iter().enumerate() {
        for &token in tokens {
            match dtype_code {
                0 => output_file.write_u8(token as u8)?,
                1 => output_file.write_u16::<LittleEndian>(token as u16)?,
                2 => output_file.write_u32::<LittleEndian>(token)?,
                _ => return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unexpected dtype_code {} at row {}", dtype_code, row_idx),
                )),
            }
        }
    }

    eprintln!("done: wrote {} frames across {} codebooks", sequence_length, num_codebooks);
    Ok(())
}
