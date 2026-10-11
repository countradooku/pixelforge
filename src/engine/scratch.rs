//! Private, checksummed scratch records with an independent recovery copy.
//! A record is published only after both writes and a verified read. Failed
//! writes leave the source in RAM. Freed extents are reused; files are removed
//! when their last record is dropped. Scratch is a cache, never a document save.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{Arc, Mutex},
};
#[cfg(any(not(unix), test))]
use std::path::PathBuf;

static SERIAL: AtomicU64 = AtomicU64::new(0);
struct Files {
    primary: File,
    recovery: File,
    end: u64,
    live: u64,
    free: Vec<(u64, u64)>,
}
pub struct Scratch {
    files: Mutex<Files>,
    #[cfg(any(not(unix), test))]
    paths: [PathBuf; 2],
    max_bytes: u64,
}
pub struct Record {
    scratch: Arc<Scratch>,
    offset: u64,
    capacity: u64,
    len: usize,
    checksum: u64,
}
fn checksum(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    })
}
impl Scratch {
    pub fn new(dir: &Path) -> io::Result<Self> {
        Self::limited(dir, u64::MAX)
    }
    fn limited(dir: &Path, max_bytes: u64) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let prefix = format!(
            "pixelforge-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Relaxed)
        );
        let paths = [
            dir.join(format!("{prefix}.scratch")),
            dir.join(format!("{prefix}.recovery")),
        ];
        let create = |p: &Path| {
            let mut o = OpenOptions::new();
            o.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                o.mode(0o600);
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                o.custom_flags(0x04000000);
            } // FILE_FLAG_DELETE_ON_CLOSE
            o.open(p)
        };
        let primary = create(&paths[0])?;
        let recovery = match create(&paths[1]) {
            Ok(f) => f,
            Err(e) => {
                let _ = std::fs::remove_file(&paths[0]);
                return Err(e);
            }
        };
        // Unlink open files on Unix: kernel ownership survives until the last
        // handle closes, including process crashes, without abandoned scratch.
        #[cfg(unix)]
        for path in &paths {
            std::fs::remove_file(path)?;
        }
        Ok(Self {
            files: Mutex::new(Files {
                primary,
                recovery,
                end: 0,
                live: 0,
                free: Vec::new(),
            }),
            #[cfg(any(not(unix), test))]
            paths,
            max_bytes,
        })
    }
    pub fn write(self: &Arc<Self>, data: &[u8]) -> io::Result<Record> {
        let mut guard = self.files.lock().unwrap();
        let f = &mut *guard;
        let len = data.len() as u64;
        let (offset, capacity) = if let Some(i) = f.free.iter().position(|(_, n)| *n >= len) {
            f.free.swap_remove(i)
        } else {
            if f.end.saturating_add(len).saturating_mul(2) > self.max_bytes {
                return Err(io::Error::other("scratch disk full"));
            }
            let offset = f.end;
            f.end += len;
            (offset, len)
        };
        let result = (|| {
            for file in [&mut f.primary, &mut f.recovery] {
                file.seek(SeekFrom::Start(offset))?;
                file.write_all(data)?;
            }
            // Read-after-write catches short writes and damaged storage before RAM is released.
            for file in [&mut f.primary, &mut f.recovery] {
                let mut check = vec![0; data.len()];
                file.seek(SeekFrom::Start(offset))?;
                file.read_exact(&mut check)?;
                if check != data {
                    return Err(io::Error::other("scratch verification failed"));
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            f.free.push((offset, capacity));
            return Err(e);
        }
        f.live += 1;
        Ok(Record {
            scratch: self.clone(),
            offset,
            capacity,
            len: data.len(),
            checksum: checksum(data),
        })
    }
}
impl Record {
    pub fn bytes(&self) -> u64 {
        self.capacity * 2
    }
    pub fn read(&self) -> io::Result<Vec<u8>> {
        let mut guard = self.scratch.files.lock().unwrap();
        let f = &mut *guard;
        let mut data = vec![0; self.len];
        for file in [&mut f.primary, &mut f.recovery] {
            if file
                .seek(SeekFrom::Start(self.offset))
                .and_then(|_| file.read_exact(&mut data))
                .is_ok()
                && checksum(&data) == self.checksum
            {
                return Ok(data);
            }
        }
        Err(io::Error::other("both scratch copies failed verification"))
    }
}
impl Drop for Record {
    fn drop(&mut self) {
        let mut f = self.scratch.files.lock().unwrap();
        f.live -= 1;
        if f.live == 0 {
            // Reclaim the file extent when the last stored tile is released.
            if f.primary.set_len(0).is_ok() && f.recovery.set_len(0).is_ok() {
                f.end = 0;
                f.free.clear();
            } else {
                f.free.push((self.offset, self.capacity));
            }
        } else {
            f.free.push((self.offset, self.capacity));
        }
    }
}
#[cfg(not(unix))]
impl Drop for Scratch {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = std::fs::remove_file(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn records_reuse_space_and_recover_corruption() {
        let s = Arc::new(Scratch::limited(&std::env::temp_dir(), 32).unwrap());
        let r = s.write(b"12345678").unwrap();
        {
            let mut f = s.files.lock().unwrap();
            f.primary.seek(SeekFrom::Start(r.offset)).unwrap();
            f.primary.write_all(b"damaged!").unwrap();
        }
        assert_eq!(r.read().unwrap(), b"12345678");
        {
            let f = s.files.lock().unwrap();
            f.recovery.set_len(0).unwrap();
        }
        assert!(r.read().is_err());
        let offset = r.offset;
        drop(r);
        let r = s.write(b"restored").unwrap();
        assert_eq!(r.offset, offset);
        assert_eq!(r.read().unwrap(), b"restored");
        let paths = s.paths.clone();
        drop(r);
        drop(s);
        assert!(paths.iter().all(|p| !p.exists()));
    }
    #[cfg(unix)]
    #[test]
    fn dropping_unlinked_scratch_preserves_replacement_paths() {
        let s = Scratch::new(&std::env::temp_dir()).unwrap();
        let paths = s.paths.clone();
        for path in &paths {
            std::fs::write(path, b"independent file").unwrap();
        }
        drop(s);
        for path in paths {
            assert_eq!(std::fs::read(&path).unwrap(), b"independent file");
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn full_disk_is_reported_without_losing_records() {
        let s = Arc::new(Scratch::limited(&std::env::temp_dir(), 16).unwrap());
        let r = s.write(b"12345678").unwrap();
        assert!(s.write(b"extra").is_err());
        assert_eq!(r.read().unwrap(), b"12345678");
    }
}
