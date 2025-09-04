use clap::Parser;
use std::fs::File;
use std::io::{self, Read, Write};
use byteorder::{LittleEndian, ReadBytesExt};
use std::path::PathBuf;

/// A program to parse and display the contents of NPQ files.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the .npq file to parse
    input_file: PathBuf,
}

#[derive(Debug)]
struct NpqHeader {
    magic: [u8; 4],
    version: u16,
    num_codebooks: u16,
    token_rate: f32,
    _orig_bitrate: f32,
    seq_len: u32,
    vocab_sizes: Vec<u32>,
    dtype_code: u8,
}

fn read_npq_header<R: Read>(reader: &mut R) -> io::Result<NpqHeader> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    if &magic != b"NPQ1" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number",
        ));
    }

    let version = reader.read_u16::<LittleEndian>()?;
    let num_codebooks = reader.read_u16::<LittleEndian>()?;
    let token_rate = reader.read_f32::<LittleEndian>()?;
    let orig_bitrate = reader.read_f32::<LittleEndian>()?;
    let seq_len = reader.read_u32::<LittleEndian>()?;

    let mut vocab_sizes = Vec::with_capacity(num_codebooks as usize);
    for _ in 0..num_codebooks {
        vocab_sizes.push(reader.read_u32::<LittleEndian>()?);
    }

    let dtype_code = reader.read_u8()?;

    Ok(NpqHeader {
        magic,
        version,
        num_codebooks,
        token_rate,
        _orig_bitrate: orig_bitrate,
        seq_len,
        vocab_sizes,
        dtype_code,
    })
}

const CROCKFORD_BASE32_ALPHABET: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

fn to_crockford_base32(mut value: u32) -> String {
    if value == 0 {
        return "0".to_string();
    }

    let mut result = Vec::new();
    while value > 0 {
        let remainder = (value % 32) as usize;
        result.push(CROCKFORD_BASE32_ALPHABET[remainder]);
        value /= 32;
    }
    result.reverse();
    String::from_utf8(result).unwrap()
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let input_path = &args.input_file;
    let output_path = input_path.with_extension("txt"); // Derive output path

    let mut input_file = File::open(input_path)?;
    let mut output_file = File::create(output_path)?;

    let header = read_npq_header(&mut input_file)?;

    // Write header to output file in new format
    let src_filename = input_path.file_name().unwrap().to_string_lossy();
    writeln!(output_file, "# npq2txt v0.1")?;
    writeln!(output_file, "# src_file: {}", src_filename)?;
    writeln!(output_file, "# magic: {}", String::from_utf8_lossy(&header.magic))?;
    writeln!(output_file, "# version: {}", header.version)?;
    writeln!(output_file, "# num_codebooks: {}", header.num_codebooks)?;
    writeln!(output_file, "# token_rate_hz: {:.4}", header.token_rate)?;
    writeln!(output_file, "# seq_len_frames: {}", header.seq_len)?;
    
    // Format vocab sizes as comma-separated list
    let vocab_sizes_str: Vec<String> = header.vocab_sizes.iter().map(|&size| size.to_string()).collect();
    writeln!(output_file, "# vocab_sizes: {}", vocab_sizes_str.join(","))?;
    
    // Map dtype_code to string representation
    let dtype_str = match header.dtype_code {
        0 => "uint8",
        1 => "uint16", 
        2 => "uint32",
        _ => "unknown"
    };
    writeln!(output_file, "# dtype: {}", dtype_str)?;
    writeln!(output_file, "")?; // Empty line before payload

    // Write payload to output file
    for _t in 0..header.seq_len as usize {
        let mut row_tokens = Vec::new();
        for i in 0..header.num_codebooks {
            let token = match header.dtype_code {
                0 => input_file.read_u8()? as u32,
                1 => input_file.read_u16::<LittleEndian>()? as u32,
                2 => input_file.read_u32::<LittleEndian>()?,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid dtype_code",
                    ))
                }
            };
            let tag = (b'a' + i as u8) as char;
            row_tokens.push(format!("{}{}", tag, to_crockford_base32(token)));
        }
        writeln!(output_file, "{}", row_tokens.join(" "))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_npq_header_valid() {
        let mut buffer = Vec::new();
        // Magic
        buffer.extend_from_slice(b"NPQ1");
        // Version
        buffer.extend_from_slice(&1u16.to_le_bytes());
        // Num Codebooks
        buffer.extend_from_slice(&8u16.to_le_bytes());
        // Token Rate
        buffer.extend_from_slice(&86.13f32.to_le_bytes());
        // Original Bitrate
        buffer.extend_from_slice(&320.0f32.to_le_bytes());
        // Sequence Length
        buffer.extend_from_slice(&100u32.to_le_bytes());
        // Vocab Sizes
        for _ in 0..8 {
            buffer.extend_from_slice(&1024u32.to_le_bytes());
        }
        // Dtype Code
        buffer.push(1);

        let mut cursor = Cursor::new(buffer);
        let header = read_npq_header(&mut cursor).unwrap();

        assert_eq!(header.magic, *b"NPQ1");
        assert_eq!(header.version, 1);
        assert_eq!(header.num_codebooks, 8);
        assert_eq!(header.token_rate, 86.13f32);
        assert_eq!(header.orig_bitrate, 320.0f32);
        assert_eq!(header.seq_len, 100);
        assert_eq!(header.vocab_sizes, vec![1024; 8]);
        assert_eq!(header.dtype_code, 1);
    }

    #[test]
    fn test_read_npq_header_invalid_magic() {
        let mut buffer = Vec::new();
        // Invalid Magic
        buffer.extend_from_slice(b"INVALID");
        let mut cursor = Cursor::new(buffer);
        let result = read_npq_header(&mut cursor);
        assert!(result.is_err());
    }
}