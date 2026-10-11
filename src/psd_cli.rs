//! Headless PSD diagnostics, migration and corpus verification.

use std::{fmt::Write as _, path::{Path, PathBuf}, process::Command, sync::Arc};

use anyhow::{Context, Result, bail, ensure};

use crate::engine::{io, memory::{self, MemoryBudget, Reservation}, metrics, psd::{PsdDocument, mapping}};

pub(crate) fn run(args: &[String]) -> Result<bool> {
    let Some(command) = args.get(1) else { return Ok(false); };
    match command.as_str() {
        "--convert-pforge" => {
            ensure!(args.len()==4,"usage: --convert-pforge input.pforge output.psd|output.psb");
            let input=Path::new(&args[2]);let output=Path::new(&args[3]);
            ensure!(mapping::is_psd(output),"Output must use .psd or .psb");
            validate_path_aliases(&[input], &[output])?;
            let (w,h,active,layers,extras)=io::load_native(input)?;
            let doc=io::native_to_document(w,h,active,layers,extras);
            let extras=io::DocExtras{paths:doc.paths,ann:doc.ann,selected:doc.selected};
            mapping::save(&doc.layers,w,h,doc.active,&extras,None,output)?;
            println!("Saved {}",output.display());
        }
        "--psd-corpus" => {
            ensure!(args.len()==3,"usage: --psd-corpus directory");
            let script=Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/psd_corpus.py");
            let status=Command::new("python3").arg(script).arg(&args[2]).arg("--binary").arg(std::env::current_exe()?).status()?;
            ensure!(status.success(),"PSD corpus acceptance failed; see pixelforge-results/report.json");
        }
        "--psd-check" => {
            ensure!(args.len()==5,"usage: --psd-check input.psd output.psd report.json");
            check(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]))?;
        }
        "--psd-metrics" => {
            ensure!(args.len()==5,"usage: --psd-metrics reference.png candidate.png report.json");
            measure(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

struct PathIdentity {
    canonical: PathBuf,
    exists: bool,
    #[cfg(unix)]
    inode: Option<(u64,u64)>,
}

impl PathIdentity {
    fn resolve(path: &Path) -> Result<Self> {
        // symlink_metadata distinguishes a new output from an existing dangling
        // symlink. The latter must fail resolution instead of bypassing checks.
        match std::fs::symlink_metadata(path) {
            Ok(_) => {
                let canonical = path.canonicalize().with_context(|| format!("Cannot resolve {}", path.display()))?;
                #[cfg(unix)]
                let inode = {
                    use std::os::unix::fs::MetadataExt;
                    let metadata = std::fs::metadata(&canonical)?;
                    Some((metadata.dev(),metadata.ino()))
                };
                Ok(Self { canonical, exists: true, #[cfg(unix)] inode })
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let filename = path.file_name().context("Output path has no filename")?;
                let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
                let canonical = parent.canonicalize().with_context(|| format!("Cannot resolve parent of {}", path.display()))?.join(filename);
                Ok(Self { canonical, exists: false, #[cfg(unix)] inode: None })
            }
            Err(error) => Err(error).with_context(|| format!("Cannot inspect {}", path.display())),
        }
    }

    fn aliases(&self, other: &Self) -> bool {
        if self.canonical == other.canonical { return true; }
        #[cfg(unix)]
        if self.inode.is_some() && self.inode == other.inode { return true; }
        // A nonexistent leaf cannot be canonicalized. Conservatively reject
        // case-only names in the same resolved directory, even on volumes
        // where two distinct files could later be created with those names.
        if (!self.exists || !other.exists) && self.canonical.parent() == other.canonical.parent() {
            if let (Some(a), Some(b)) = (self.canonical.file_name(), other.canonical.file_name()) {
                return a.eq_ignore_ascii_case(b);
            }
        }
        false
    }
}

/// Reads may intentionally name the same file (for example, a metric's exact
/// comparison). Every output must be distinct from every input and output.
/// Resolve the complete set before a codec, export or report can write anything.
fn validate_path_aliases(inputs: &[&Path], outputs: &[&Path]) -> Result<()> {
    let inputs_resolved: Vec<_> = inputs.iter().map(|path| PathIdentity::resolve(path)).collect::<Result<_>>()?;
    let outputs_resolved: Vec<_> = outputs.iter().map(|path| PathIdentity::resolve(path)).collect::<Result<_>>()?;
    for (index, output) in outputs_resolved.iter().enumerate() {
        for (input_index, input) in inputs_resolved.iter().enumerate() {
            ensure!(!output.aliases(input), "Output {} aliases input {}; no files were written", outputs[index].display(), inputs[input_index].display());
        }
        for (previous_index, previous) in outputs_resolved[..index].iter().enumerate() {
            ensure!(!output.aliases(previous), "Outputs {} and {} alias the same file; no files were written", outputs[index].display(), outputs[previous_index].display());
        }
    }
    Ok(())
}

pub(crate) fn quote(value:&str)->String {
    let mut output=String::from("\"");
    for c in value.chars() {
        match c {'"'=>output.push_str("\\\""),'\\'=>output.push_str("\\\\"),'\n'=>output.push_str("\\n"),'\r'=>output.push_str("\\r"),'\t'=>output.push_str("\\t"),c if c<' '=>{let _=write!(output,"\\u{:04x}",c as u32);},c=>output.push(c)}
    }
    output.push('"');output
}

fn check(input:&Path,output:&Path,report:&Path)->Result<()> {
    let rendering = report.with_extension("png");
    validate_path_aliases(&[input], &[output,report,&rendering])?;
    let started=std::time::Instant::now();
    let limits=mapping::limits();
    let tree=PsdDocument::read(input,&limits)?;
    let read_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=std::time::Instant::now();
    let composite=tree.decode_composite(&limits)?;
    let decoded_composite_bytes=composite.len();drop(composite);
    // Independently validate every channel's stream, not only streams used by
    // the editable mapper. The mask rectangle controls its channel dimensions.
    let mut decoded_channels=0usize;
    for layer in &tree.layer_mask.layers {
        for channel in &layer.channels {
            let (w,h)=channel_dimensions(layer,channel.id)?;
            drop(crate::engine::psd::compression::decode(channel.encoded.as_slice(),w,h,tree.header.depth,tree.header.version,&limits)?);
            decoded_channels+=1;
        }
    }
    let decode_ms=started.elapsed().as_secs_f64()*1000.0;
    let raw_encoding = tree.encode(&limits)?;
    let exact_file = std::fs::read(input)? == raw_encoding;
    let raw_reopened = PsdDocument::parse(raw_encoding, &limits)?;
    let structural_equal = tree == raw_reopened;
    let inspections = crate::engine::psd::inspect::inspect(&tree);
    let descriptor_valid = inspections.iter().all(|row| row.status != "malformed");
    let inspection_json = inspections.iter().map(|row| format!(
        "{{\"location\":{},\"key\":{},\"status\":{},\"descriptors\":{},\"engine_data\":{},\"message\":{}}}",
        quote(&row.location), quote(&row.key), quote(row.status), row.descriptors, row.engine_data, quote(&row.message)
    )).collect::<Vec<_>>().join(",");
    let mapping_result = mapping::from_tree(tree.clone());
    let (mapping_supported, error, render_ms, mapped) = match mapping_result {
        Ok(doc) => {
            let started = std::time::Instant::now();
            io::export(&doc.layers, doc.w, doc.h, &rendering)?;
            let render_ms = started.elapsed().as_secs_f64() * 1000.0;
            let extras = io::DocExtras { paths: doc.paths, ann: doc.ann, selected: doc.selected };
            match mapping::to_tree(&doc.layers, doc.w, doc.h, doc.active, &extras, doc.psd.as_deref(), tree.header.version) {
                Ok(mapped) => (true, String::new(), render_ms, mapped),
                Err(error) => (false, format!("Writer: {error:#}"), render_ms, tree.clone()),
            }
        }
        Err(error) => (false, format!("{error:#}"), 0.0, tree.clone()),
    };
    let started = std::time::Instant::now();
    mapped.write(output, &limits)?;
    let write_ms = started.elapsed().as_secs_f64() * 1000.0;
    let reopened = PsdDocument::read(output, &limits)?;
    let mapped_structural_equal = mapped == reopened;
    let roundtrip_kind = if mapping_supported { "document-model" } else { "raw-preservation-only" };
    let keys: Vec<String> = tree.layer_mask.layers.iter().flat_map(|l| l.additional_info.iter())
        .map(|b| quote(&String::from_utf8_lossy(&b.key))).collect();
    let text = format!("{{
  \"structural_equal\":{structural_equal},
  \"exact_file\":{exact_file},
  \"mapped_structural_equal\":{mapped_structural_equal},
  \"roundtrip_kind\":{},
  \"mapping_supported\":{mapping_supported},
  \"mapping_error\":{},
  \"descriptor_valid\":{descriptor_valid},
  \"inspections\":[{inspection_json}],
  \"mode\":{},
  \"depth\":{},
  \"layers\":{},
  \"decoded_channels\":{decoded_channels},
  \"decoded_composite_bytes\":{decoded_composite_bytes},
  \"read_ms\":{read_ms},
  \"decode_ms\":{decode_ms},
  \"write_ms\":{write_ms},
  \"render_ms\":{render_ms},
  \"layer_block_keys\":[{}]
}}\n", quote(roundtrip_kind), quote(&error), quote(mapping::mode_name(tree.header.color_mode)), tree.header.depth, tree.layer_mask.layers.len(), keys.join(","));
    std::fs::write(report, text)?;
    ensure!(structural_equal && exact_file && mapped_structural_equal, "PSD structural/byte preservation failed");
    ensure!(descriptor_valid, "PSD descriptor inspection failed; see inspection rows");
    ensure!(mapping_supported, "PSD editable mapping failed: {error}");
    Ok(())
}

fn channel_dimensions(layer:&crate::engine::psd::LayerRecord,id:i16)->Result<(u32,u32)> {
    if id != -2 && id != -3 {return Ok(layer.dimensions()?);}
    let mask = crate::engine::psd::masks::MaskMetadata::parse(&layer.mask, layer.channels.iter().any(|channel| channel.id == -3))?
        .ok_or_else(|| anyhow::anyhow!("Mask channel without mask metadata"))?;
    let bounds = mask.channel(id)?.bounds;
    Ok((u32::try_from(i64::from(bounds[3])-i64::from(bounds[1]))?, u32::try_from(i64::from(bounds[2])-i64::from(bounds[0]))?))
}

const MAX_METRIC_PIXELS: u64 = 1 << 28; // The 16384 × 16384 acceptance image.
const MAX_METRIC_DIMENSION: u32 = 300_000; // PSB's dimension ceiling.
const METRIC_DECODER_SCRATCH: u64 = 64 << 20;

struct MetricImage {
    pixels: image::RgbaImage,
    // Keep the charge until after the pixel buffer is dropped.
    _memory: Reservation,
}

fn reserve_metric_memory(budget: &Arc<MemoryBudget>, bytes: u64) -> Result<Reservation> {
    // Charge atomically before checking the shared limit, so concurrent
    // reservations cannot both approve the same remaining capacity.
    let reservation = budget.track_cpu(bytes);
    ensure!(budget.stats().used <= budget.limit(), "Composite metrics exceed the shared memory budget (requested {bytes} bytes)");
    Ok(reservation)
}

fn metric_rgba_bytes(width: u32, height: u32) -> Result<u64> {
    ensure!(width > 0 && height > 0, "Empty composite");
    ensure!(width <= MAX_METRIC_DIMENSION && height <= MAX_METRIC_DIMENSION, "Composite dimensions exceed the metric limit");
    let pixels = u64::from(width).checked_mul(u64::from(height)).context("Composite dimensions overflow")?;
    ensure!(pixels <= MAX_METRIC_PIXELS, "Composite exceeds the 268,435,456-pixel metric limit");
    Ok(pixels * 4)
}

fn metric_decoder_limits(max_alloc: u64) -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_METRIC_DIMENSION);
    limits.max_image_height = Some(MAX_METRIC_DIMENSION);
    limits.max_alloc = Some(max_alloc);
    limits
}

fn load_metric_image(path: &Path, budget: &Arc<MemoryBudget>) -> Result<MetricImage> {
    use image::ImageDecoder;
    // PNG retains this internal scratch limit after construction. The pixel
    // buffer is allocated separately below, after dimensions and budget are
    // checked; ImageReader::decode would instead apply its 512 MiB default to
    // that buffer. There is no dimensions-only pass or second file read.
    let scratch_memory = reserve_metric_memory(budget, METRIC_DECODER_SCRATCH)?;
    let mut reader = image::ImageReader::open(path)?;
    let is_png = reader.format() == Some(image::ImageFormat::Png);
    reader.limits(metric_decoder_limits(METRIC_DECODER_SCRATCH));
    let mut decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    let rgba_bytes = metric_rgba_bytes(width, height)?;
    let native_bytes = decoder.total_bytes();
    ensure!(native_bytes <= MAX_METRIC_PIXELS * 16, "Composite decoder output exceeds the metric allocation limit");
    ensure!(usize::try_from(native_bytes).is_ok() && usize::try_from(rgba_bytes).is_ok(), "Composite allocation exceeds the platform address space");
    // Non-PNG decoders may retain a native frame in addition to the returned
    // DynamicImage. PNG decodes directly into the supplied output buffer.
    let codec_memory = if is_png { None } else { Some(reserve_metric_memory(budget, native_bytes)?) };
    decoder.set_limits(metric_decoder_limits(native_bytes + METRIC_DECODER_SCRATCH))?;
    let native_memory = reserve_metric_memory(budget, native_bytes)?;
    let decoded = image::DynamicImage::from_decoder(decoder)?;
    drop(codec_memory);
    drop(scratch_memory);
    if decoded.color() == image::ColorType::Rgba8 {
        return Ok(MetricImage { pixels: decoded.into_rgba8(), _memory: native_memory });
    }
    // RGB, grayscale and 16-bit input conversion temporarily keeps both
    // native and RGBA buffers alive; neither allocation is left uncharged.
    let rgba_memory = reserve_metric_memory(budget, rgba_bytes)?;
    let pixels = decoded.into_rgba8();
    drop(native_memory);
    Ok(MetricImage { pixels, _memory: rgba_memory })
}

fn metric_scratch_bytes(width: usize, rows: usize) -> Result<u64> {
    // SSIM peaks at eleven f32 planes; two RGB strips and a byte mask add
    // seven bytes per pixel. Round 51 up to 64 and include the small kernel.
    (width as u64).checked_mul(rows as u64).and_then(|n| n.checked_mul(64)).and_then(|n| n.checked_add(4096)).context("Composite metric scratch size overflow")
}

fn measure(reference:&Path,candidate:&Path,report:&Path)->Result<()> {
    validate_path_aliases(&[reference,candidate], &[report])?;
    let budget = memory::global();
    let a_image = load_metric_image(reference, budget)?;
    let b_image = load_metric_image(candidate, budget)?;
    let a = &a_image.pixels;
    let b = &b_image.pixels;
    ensure!(a.dimensions()==b.dimensions(),"Composite dimensions differ");
    let (w,h)=(a.width() as usize,a.height() as usize);
    ensure!(w>0 && h>0,"Empty composite");
    let exact=a.as_raw()==b.as_raw();
    let differing_pixels=a.pixels().zip(b.pixels()).filter(|(a,b)|a!=b).count();
    let max_error=a.as_raw().iter().zip(b.as_raw()).map(|(&a,&b)|a.abs_diff(b)).max().unwrap_or(0);
    let alpha_exact=a.pixels().zip(b.pixels()).all(|(a,b)|a[3]==b[3]);
    let (ssim,delta)=if exact {(1.0,0.0)} else {
        // Full-resolution strips with the SSIM kernel halo. This bounds the
        // metric's scratch buffers without downsampling the evidence.
        let mut sums=(0.0,0.0,0usize);
        for top in (0..h).step_by(128) {
            let bottom=(top+128).min(h);let start=top.saturating_sub(5);let end=(bottom+5).min(h);
            let _scratch_memory = reserve_metric_memory(budget, metric_scratch_bytes(w, end-start)?)?;
            let to_rgb=|image:&image::RgbaImage|->Vec<[u8;3]>{
                image.as_raw()[start*w*4..end*w*4].chunks_exact(4).map(|p| {
                    // Compare visible colour on white; alpha equality is a
                    // separate mandatory gate rather than lost in this matte.
                    [0,1,2].map(|c|((u32::from(p[c])*u32::from(p[3])+255*(255-u32::from(p[3]))+127)/255) as u8)
                }).collect()
            };
            let aa=to_rgb(a);let bb=to_rgb(b);let mut mask=vec![0;w*(end-start)];
            mask[(top-start)*w..(bottom-start)*w].fill(255);let n=(bottom-top)*w;
            sums.0+=metrics::ssim(&aa,&bb,w,end-start,Some(&mask))*n as f64;
            sums.1+=metrics::mean_delta_e(&aa,&bb,Some(&mask))*n as f64;sums.2+=n;
        }
        (sums.0/sums.2 as f64,sums.1/sums.2 as f64)
    };
    ensure!(ssim.is_finite() && delta.is_finite(),"Non-finite fidelity score");
    let passed=alpha_exact && ssim>=0.995 && delta<=0.5;
    std::fs::write(report,format!("{{\"exact\":{exact},\"alpha_exact\":{alpha_exact},\"differing_pixels\":{differing_pixels},\"max_channel_error\":{max_error},\"ssim\":{ssim},\"mean_delta_e00\":{delta},\"passed\":{passed}}}\n"))?;
    if !passed {bail!("Composite fidelity failed: SSIM={ssim:.6}, mean DeltaE00={delta:.6}, max error={max_error}");}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check, load_metric_image, measure, metric_rgba_bytes, reserve_metric_memory, run, validate_path_aliases, METRIC_DECODER_SCRATCH};
    use std::{fs, path::{Path, PathBuf}, sync::atomic::{AtomicU64, Ordering}};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir().join(format!("pixelforge-cli-alias-{}-{stamp}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self, name: &str) -> PathBuf { self.0.join(name) }

        fn input(&self, name: &str) -> PathBuf {
            let path = self.path(name);
            fs::write(&path, b"source must survive before decoding").unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    fn assert_source_untouched(path: &Path) {
        assert_eq!(fs::read(path).unwrap(), b"source must survive before decoding");
    }

    #[test]
    fn json_quotes_control_characters_and_paths() {
        assert_eq!(super::quote("a\n\t\u{0001}\\\""),"\"a\\n\\t\\u0001\\\\\\\"\"");
    }

    #[test]
    fn check_rejects_source_as_output_or_report_before_decoding() {
        let directory = TestDirectory::new();
        let input = directory.input("source.psd");
        let output = directory.path("roundtrip.psd");
        let report = directory.path("report.json");
        for result in [check(&input, &input, &report), check(&input, &output, &input)] {
            assert!(result.unwrap_err().to_string().contains("aliases input"));
            assert_source_untouched(&input);
            assert!(!output.exists());
            assert!(!report.exists());
            assert!(!report.with_extension("png").exists());
        }
    }

    #[test]
    fn check_rejects_collisions_among_every_output_before_decoding() {
        let directory = TestDirectory::new();
        let input = directory.input("source.psd");
        let report = directory.path("report.json");
        let rendering = report.with_extension("png");
        let output = directory.path("roundtrip.psd");
        for result in [check(&input, &report, &report), check(&input, &rendering, &report), check(&input, &output, &rendering)] {
            assert!(result.unwrap_err().to_string().contains("alias the same file"));
            assert_source_untouched(&input);
            assert!(!output.exists());
            assert!(!report.exists());
            assert!(!rendering.exists());
        }
    }

    #[test]
    fn check_rejects_derived_rendering_aliasing_input_before_decoding() {
        let directory = TestDirectory::new();
        let input = directory.input("source.png");
        let output = directory.path("roundtrip.psd");
        let report = directory.path("source.json");
        assert!(check(&input, &output, &report).unwrap_err().to_string().contains("aliases input"));
        assert_source_untouched(&input);
        assert!(!output.exists());
        assert!(!report.exists());
    }

    #[test]
    fn metric_inputs_may_alias_but_the_report_must_be_distinct() {
        let directory = TestDirectory::new();
        let reference = directory.input("reference.png");
        let candidate = directory.input("candidate.png");
        let report = directory.path("report.json");
        validate_path_aliases(&[&reference, &reference], &[&report]).unwrap();
        for report in [&reference, &candidate] {
            assert!(measure(&reference, &candidate, report).unwrap_err().to_string().contains("aliases input"));
            assert_source_untouched(&reference);
            assert_source_untouched(&candidate);
        }
        // Exercise the full metric path too: identical read inputs are valid.
        image::RgbaImage::from_pixel(1, 1, image::Rgba([10, 20, 30, 255])).save(&reference).unwrap();
        measure(&reference, &reference, &report).unwrap();
        assert!(fs::read_to_string(&report).unwrap().contains("\"passed\":true"));
    }

    #[test]
    fn canonical_parent_resolution_catches_new_output_aliases() {
        let directory = TestDirectory::new();
        let output = directory.path("report.json");
        let equivalent = directory.path("./report.json");
        assert!(validate_path_aliases(&[], &[&output, &equivalent]).unwrap_err().to_string().contains("alias the same file"));
        assert!(!output.exists());
    }

    #[test]
    fn missing_output_names_are_conservatively_compared_without_ascii_case() {
        let directory = TestDirectory::new();
        let input = directory.input("source.psd");
        let output = directory.path("roundtrip.psd");
        let report = directory.path("report.PNG");
        // report.PNG would alias its derived report.png on common macOS
        // volumes. This test also exercises the policy on case-sensitive CI.
        assert!(check(&input, &output, &report).unwrap_err().to_string().contains("alias the same file"));
        assert_source_untouched(&input);
        assert!(!output.exists());
        assert!(!report.exists());
        assert!(!report.with_extension("png").exists());
    }

    #[test]
    fn metric_dimensions_admit_the_full_acceptance_image_and_bound_larger_inputs() {
        assert_eq!(metric_rgba_bytes(16_384, 16_384).unwrap(), 1 << 30);
        assert!(metric_rgba_bytes(16_384, 16_385).unwrap_err().to_string().contains("pixel metric limit"));
        assert!(metric_rgba_bytes(300_001, 1).unwrap_err().to_string().contains("dimensions"));
        assert!(metric_rgba_bytes(u32::MAX, u32::MAX).is_err());
        assert!(metric_rgba_bytes(0, 1).is_err());
    }

    #[test]
    fn metric_decoding_tracks_retained_pixels_and_native_conversion_allocations() {
        use crate::engine::memory::MemoryBudget;
        let directory = TestDirectory::new();
        let rgba = directory.path("rgba.png");
        let rgb = directory.path("rgb.png");
        let wide = directory.path("rgba16.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255])).save(&rgba).unwrap();
        image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30])).save(&rgb).unwrap();
        image::ImageBuffer::<image::Rgba<u16>, Vec<u16>>::from_pixel(2, 2, image::Rgba([u16::MAX, 0, 0, u16::MAX])).save(&wide).unwrap();
        let budget = MemoryBudget::new(METRIC_DECODER_SCRATCH + 128, directory.0.clone(), 0);
        let a = load_metric_image(&rgba, &budget).unwrap();
        assert_eq!(budget.stats().used, 16);
        let b = load_metric_image(&rgb, &budget).unwrap();
        assert_eq!(a.pixels, b.pixels);
        assert_eq!(budget.stats().used, 32);
        let c = load_metric_image(&wide, &budget).unwrap();
        assert_eq!(c.pixels.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(budget.stats().used, 48);
        assert!(budget.stats().peak >= METRIC_DECODER_SCRATCH + 32 + 32);
        drop((a, b, c));
        assert_eq!(budget.stats().used, 0);
    }

    #[test]
    fn metric_budget_refusal_releases_reservations_before_pixel_allocation() {
        use crate::engine::memory::MemoryBudget;
        let directory = TestDirectory::new();
        let path = directory.path("rgba.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255])).save(&path).unwrap();
        let budget = MemoryBudget::new(METRIC_DECODER_SCRATCH + 4, directory.0.clone(), 0);
        let error = load_metric_image(&path, &budget).err().expect("pixel allocation must exceed this budget");
        assert!(error.to_string().contains("shared memory budget"));
        assert_eq!(budget.stats().used, 0);
        let held = reserve_metric_memory(&budget, budget.limit()).unwrap();
        assert!(reserve_metric_memory(&budget, 1).is_err());
        assert_eq!(budget.stats().used, budget.limit());
        drop(held);
        assert_eq!(budget.stats().used, 0);
    }

    #[cfg(unix)]
    #[test]
    fn existing_hard_links_are_rejected_for_conversion_and_outputs() {
        let directory = TestDirectory::new();
        let input = directory.input("source.pforge");
        let output = directory.path("output.psd");
        fs::hard_link(&input, &output).unwrap();
        let args = ["pixelforge".to_string(), "--convert-pforge".to_string(), input.to_string_lossy().into_owned(), output.to_string_lossy().into_owned()];
        assert!(run(&args).unwrap_err().to_string().contains("aliases input"));
        assert_source_untouched(&input);
        assert_source_untouched(&output);
        assert!(validate_path_aliases(&[], &[&input, &output]).unwrap_err().to_string().contains("alias the same file"));
        // Aliased reads remain safe, including inode-only aliases.
        validate_path_aliases(&[&input, &output], &[&directory.path("report.json")]).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_files_and_new_outputs_in_symlink_parents_are_resolved() {
        use std::os::unix::fs::symlink;
        let directory = TestDirectory::new();
        let input = directory.input("source.psd");
        let output = directory.path("output.psd");
        symlink(&input, &output).unwrap();
        assert!(validate_path_aliases(&[&input], &[&output]).unwrap_err().to_string().contains("aliases input"));
        let parent_alias = directory.path("linked-parent");
        symlink(&directory.0, &parent_alias).unwrap();
        let report = directory.path("report.json");
        let aliased_report = parent_alias.join("report.json");
        assert!(validate_path_aliases(&[], &[&report, &aliased_report]).unwrap_err().to_string().contains("alias the same file"));
        assert_source_untouched(&input);
        assert!(!report.exists());
    }

    #[cfg(unix)]
    #[test]
    fn dangling_output_symlink_fails_without_writing_its_target() {
        use std::os::unix::fs::symlink;
        let directory = TestDirectory::new();
        let target = directory.path("missing.psd");
        let output = directory.path("output.psd");
        symlink(&target, &output).unwrap();
        assert!(validate_path_aliases(&[], &[&output]).unwrap_err().to_string().contains("Cannot resolve"));
        assert!(!target.exists());
    }
}
