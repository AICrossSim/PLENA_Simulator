use std::path::{Path, PathBuf};

pub(crate) fn read_hbm_preload(path: &Path) -> Vec<u8> {
    std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read HBM preload file {path:?}: {err}"))
}

pub(crate) fn read_fpsram_preload(path: &Path) -> Vec<u8> {
    std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read FP SRAM preload file {path:?}: {err}"))
}

pub(crate) fn read_optional_intsram_preload(path: Option<PathBuf>) -> Option<Vec<u8>> {
    path.map(|path| {
        std::fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read INT SRAM preload file {path:?}: {err}"))
    })
}

pub(crate) fn read_optional_vram_preload(path: Option<PathBuf>) -> Option<Vec<u8>> {
    path.map(|path| {
        std::fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read VRAM preload file {path:?}: {err}"))
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        read_fpsram_preload, read_hbm_preload, read_optional_intsram_preload,
        read_optional_vram_preload,
    };

    fn write_temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("plena-preload-test-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn required_preload_readers_return_file_bytes() {
        let hbm = write_temp_file("hbm", &[1, 2, 3]);
        let fpsram = write_temp_file("fpsram", &[4, 5]);

        assert_eq!(read_hbm_preload(&hbm), vec![1, 2, 3]);
        assert_eq!(read_fpsram_preload(&fpsram), vec![4, 5]);

        std::fs::remove_file(hbm).unwrap();
        std::fs::remove_file(fpsram).unwrap();
    }

    #[test]
    fn optional_preload_readers_return_none_without_path() {
        assert_eq!(read_optional_intsram_preload(None), None);
        assert_eq!(read_optional_vram_preload(None), None);
    }

    #[test]
    fn optional_preload_readers_return_bytes_when_path_is_present() {
        let intsram = write_temp_file("intsram", &[9, 8]);
        let vram = write_temp_file("vram", &[7, 6, 5]);

        assert_eq!(
            read_optional_intsram_preload(Some(intsram.clone())),
            Some(vec![9, 8])
        );
        assert_eq!(
            read_optional_vram_preload(Some(vram.clone())),
            Some(vec![7, 6, 5])
        );

        std::fs::remove_file(intsram).unwrap();
        std::fs::remove_file(vram).unwrap();
    }
}
