use std::path::Path;

use crate::op;

pub(crate) fn read_opcode_words(path: &Path) -> Vec<u32> {
    let op_file = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read opcode file {path:?}: {err}"));
    parse_opcode_words(&op_file)
}

fn parse_opcode_words(contents: &str) -> Vec<u32> {
    contents
        .split_whitespace()
        .map(|tok| {
            u32::from_str_radix(tok.trim_start_matches("0x"), 16)
                .unwrap_or_else(|err| panic!("failed to parse opcode hex token {tok:?}: {err}"))
        })
        .collect()
}

pub(crate) fn decode_program(words: Vec<u32>) -> Vec<op::Opcode> {
    words.into_iter().map(op::Opcode::decode).collect()
}

#[cfg(test)]
mod tests {
    use super::{decode_program, parse_opcode_words, read_opcode_words};
    use crate::op::Opcode;

    fn write_temp_file(name: &str, contents: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("plena-program-test-{name}-{}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn parse_opcode_words_accepts_hex_prefixes_and_whitespace() {
        let words = parse_opcode_words("0x01 02\n0x00000003\t04");

        assert_eq!(words, vec![1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "failed to parse opcode hex token \"zz\"")]
    fn parse_opcode_words_panics_with_bad_token_context() {
        parse_opcode_words("0x01 zz");
    }

    #[test]
    fn read_opcode_words_reads_file_contents() {
        let path = write_temp_file("read", "0x00 0x32");

        let words = read_opcode_words(&path);

        assert_eq!(words, vec![0x00, 0x32]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn decode_program_maps_words_through_opcode_decoder() {
        let decoded = decode_program(vec![0x00, 0x32]);

        assert!(matches!(decoded[0], Opcode::Invalid));
        assert!(matches!(decoded[1], Opcode::C_BREAK));
    }
}
