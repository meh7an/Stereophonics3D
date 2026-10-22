//! Reading the G3 frame counter back from an image.
//!
//! Each probe is one cell of the counter, most significant bit first; its centre pixel is
//! classified as the probe's `on` or `off` colour, whichever is nearer.

use s3d_shared::capture::{Image, ProbeRecord};

use crate::{
    Error, Result,
    image::{Rgb, ndc_to_pixel, pixel},
};

/// Decodes the counter shown in `image`.
///
/// # Errors
///
/// Returns [`Error::NotApplicable`] when there are no probes or a probe lies outside the
/// image.
pub fn decode(image: &Image, probes: &[ProbeRecord]) -> Result<u32> {
    if probes.is_empty() {
        return Err(not_applicable("the capture has no counter probes".into()));
    }
    let mut sorted: Vec<&ProbeRecord> = probes.iter().collect();
    sorted.sort_by_key(|probe| probe.bit);

    let mut value = 0;
    for probe in sorted {
        let (x, y) = ndc_to_pixel(probe.ndc, image.width, image.height);
        let inside = (0.0..f64::from(image.width)).contains(&x)
            && (0.0..f64::from(image.height)).contains(&y);
        if !inside {
            return Err(not_applicable(format!(
                "probe {} at {:?} lies outside the image",
                probe.bit, probe.ndc
            )));
        }
        let color = pixel(image, x.floor() as u32, y.floor() as u32);
        let set = distance(color, probe.on) < distance(color, probe.off);
        value = (value << 1) | u32::from(set);
    }
    Ok(value)
}

fn distance(a: Rgb, b: Rgb) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, y)| u32::from(x.abs_diff(y)))
        .sum()
}

fn not_applicable(reason: String) -> Error {
    Error::NotApplicable {
        test: "counter",
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ON: Rgb = [255, 255, 255];
    const OFF: Rgb = [70, 70, 70];

    fn probes() -> Vec<ProbeRecord> {
        (0..4)
            .map(|bit| ProbeRecord {
                bit,
                ndc: [-0.75 + bit as f32 * 0.5, 0.0],
                on: ON,
                off: OFF,
            })
            .collect()
    }

    /// An 8x2 image whose probe pixels (row 1, columns 1, 3, 5, 7) show `bits`.
    fn image(bits: [bool; 4]) -> Image {
        let (width, height) = (8usize, 2usize);
        let mut rgba = vec![0; width * height * 4];
        for (bit, set) in bits.into_iter().enumerate() {
            let offset = (width + 1 + bit * 2) * 4;
            rgba[offset..offset + 3].copy_from_slice(if set { &ON } else { &OFF });
        }
        Image {
            width: width as u32,
            height: height as u32,
            rgba,
        }
    }

    #[test]
    fn test_decode_reads_msb_first() {
        assert_eq!(
            decode(&image([true, false, true, true]), &probes()).expect("decode"),
            0b1011
        );
    }

    #[test]
    fn test_decode_independent_of_probe_order() {
        let mut reversed = probes();
        reversed.reverse();
        assert_eq!(
            decode(&image([false, true, false, false]), &reversed).expect("decode"),
            0b0100
        );
    }

    #[test]
    fn test_decode_without_probes_is_not_applicable() {
        assert!(decode(&image([false; 4]), &[]).is_err());
    }
}
