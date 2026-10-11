//! OS measurements outside the timed regions. Peaks are process lifetime
//! high-water marks; page-ins include file-backed reads, not just swap.

#[derive(Clone, Copy, Debug, Default)]
struct Usage {
    resident: Option<u64>,
    peak_resident: Option<u64>,
    pageins: Option<u64>,
}

pub struct Report {
    start: Usage,
}

impl Report {
    pub fn start() -> Self {
        let memory = crate::engine::memory::global().stats();
        println!("  budget before: {}", memory.summary());
        let start = usage();
        println!(
            "  memory before: RSS {} MiB; depth 8 bits/channel",
            mib(start.resident)
        );
        Self { start }
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        let memory = crate::engine::memory::global().stats();
        println!(
            "  budget after: {}; peak tracked {:.1} MiB",
            memory.summary(),
            memory.peak as f64 / 1048576.
        );
        if let Some(e) = memory.fault {
            println!("  scratch warning: {e}");
        }
        let end = usage();
        let pageins = self
            .start
            .pageins
            .zip(end.pageins)
            .map(|(a, b)| b.saturating_sub(a));
        println!(
            "  memory after: RSS {} MiB; peak RSS {} MiB (process lifetime); page-ins {} (includes file reads)",
            mib(end.resident),
            mib(end.peak_resident),
            pageins.map_or_else(|| "unavailable".into(), |n| n.to_string())
        );
    }
}

fn mib(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "unavailable".into(),
        |n| format!("{:.1}", n as f64 / 1_048_576.0),
    )
}

#[cfg(target_os = "macos")]
fn usage() -> Usage {
    use std::ffi::{c_int, c_long, c_void};
    // Layouts from the macOS SDK's sys/resource.h and sys/_types/_timeval.h.
    // RUSAGE_INFO_V0 has been available since macOS 10.9.
    #[repr(C)]
    #[derive(Default)]
    struct Info {
        uuid: [u8; 16],
        user_time: u64,
        system_time: u64,
        pkg_idle_wkups: u64,
        interrupt_wkups: u64,
        pageins: u64,
        wired_size: u64,
        resident_size: u64,
        phys_footprint: u64,
        proc_start_abstime: u64,
        proc_exit_abstime: u64,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Timeval {
        sec: c_long,
        usec: c_int,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Rusage {
        user_time: Timeval,
        system_time: Timeval,
        // ru_maxrss followed by the thirteen other integral counters.
        counters: [c_long; 14],
    }
    unsafe extern "C" {
        fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
        fn getrusage(who: c_int, buffer: *mut Rusage) -> c_int;
    }
    let mut info = Info::default();
    let mut ru = Rusage::default();
    // SAFETY: both writable buffers have the C ABI layout and size expected
    // for flavor 0 / RUSAGE_SELF. The calls are synchronous and cannot keep
    // the pointers. We only inspect results when the call succeeds.
    let has_info = unsafe {
        proc_pid_rusage(
            std::process::id() as c_int,
            0,
            (&mut info as *mut Info).cast(),
        ) == 0
    };
    // SAFETY: see above; who=0 is RUSAGE_SELF on macOS.
    let has_ru = unsafe { getrusage(0, &mut ru) == 0 };
    Usage {
        resident: has_info.then_some(info.resident_size),
        peak_resident: has_ru.then(|| u64::try_from(ru.counters[0]).ok()).flatten(),
        pageins: has_info.then_some(info.pageins),
    }
}

#[cfg(target_os = "linux")]
fn usage() -> Usage {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    Usage {
        resident: status_bytes(&text, "VmRSS:"),
        peak_resident: status_bytes(&text, "VmHWM:"),
        pageins: None,
    }
}

#[cfg(any(target_os = "linux", test))]
fn status_bytes(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find_map(|l| l.strip_prefix(key))?;
    let mut fields = line.split_whitespace();
    let kb = fields.next()?.parse::<u64>().ok()?;
    (fields.next()? == "kB")
        .then(|| kb.checked_mul(1024))
        .flatten()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn usage() -> Usage {
    Usage::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_status_reads_rss_and_peak_with_correct_units() {
        let text = "Name:\tpixelforge\nVmHWM:\t2048 kB\nVmRSS:\t1024 kB\n";
        assert_eq!(status_bytes(text, "VmRSS:"), Some(1_048_576));
        assert_eq!(status_bytes(text, "VmHWM:"), Some(2_097_152));
    }

    #[test]
    fn missing_or_bad_measurements_are_unavailable_not_zero() {
        for text in [
            "",
            "VmRSS: bad kB",
            "VmRSS: 12 bytes",
            "VmRSS: 18446744073709551615 kB",
        ] {
            assert_eq!(status_bytes(text, "VmRSS:"), None);
        }
        assert_eq!(mib(None), "unavailable");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_reports_valid_process_memory() {
        let u = usage();
        assert!(u.resident.is_some_and(|n| n > 0));
        assert!(u.peak_resident.is_some_and(|n| n > 0));
        assert!(u.pageins.is_some());
    }
}
