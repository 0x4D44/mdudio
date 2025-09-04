# GEMINI Context for txt2npq tool

## User Request

The user wants to create a tool named `txt2npq` in the `C:\language\mdudio\txt2npq` directory. This tool will convert a text file to a `.npq` file, which is the reverse of the `npq2txt` tool.

## Assumed Input Text File Format

The input text file is expected to have the following format:

- Key-value pairs for the header fields, each on a new line.
- A "Payload:" marker.
- The payload as a series of lines, each starting with "T=..." followed by a list of tokens.

### Example:

```
Magic: NPQ1
Version: 1
Num Codebooks (K): 8
Token Rate (fps): 86.13
Original Bitrate (kbps): 320.0
Sequence Length (T): 100
Vocab Sizes: [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
Dtype Code: 1

Payload:
T=0: [1, 2, 3, 4, 5, 6, 7, 8]
T=1: [9, 10, 11, 12, 13, 14, 15, 16]
```

## Implementation Plan

1.  **Initialize a new Cargo project** in `C:\language\mdudio\txt2npq` (or use the existing one).
2.  **Add dependencies** to `Cargo.toml`:
    *   `clap` for command-line argument parsing.
    *   `byteorder` for writing binary data.
3.  **Implement the main logic** in `src/main.rs`:
    *   Parse command-line arguments to get the input text file path and the output `.npq` file path.
    *   Read and parse the input text file to extract the header and payload.
    *   Write the header and payload to the output `.npq` file in the correct binary format.
4.  **Add comprehensive unit tests** to verify the text parsing and binary writing logic.
5.  **Create a `GEMINI.md` file** in the new project directory to document the project for future interactions.
