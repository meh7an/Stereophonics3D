//! Pixel probes that read the G3 frame counter back from a captured image.
//!
//! The verifier takes this over in M2.4; until then `--self-check` uses it to prove the
//! counter is readable from a capture.

use anyhow::{Result, ensure};

use crate::scene::Probe;

/// Decodes the counter from RGBA8 `pixels` by classifying each probe's centre pixel as
/// its `on` or `off` colour, most significant bit first.
///
/// # Errors
///
/// Returns an error when the image size does not match or a probe lies outside it.
pub fn decode_counter(pixels: &[u8], width: u32, height: u32, probes: &[Probe]) -> Result<u32> {
    ensure!(
        pixels.len() == width as usize * height as usize * 4,
        "image is not {width}x{height} RGBA8"
    );
    let mut sorted: Vec<&Probe> = probes.iter().collect();
    sorted.sort_by_key(|probe| probe.bit);

    let mut value = 0;
    for probe in sorted {
        let x = ((probe.ndc[0] * 0.5 + 0.5) * width as f32).floor();
        let y = ((probe.ndc[1] * 0.5 + 0.5) * height as f32).floor();
        ensure!(
            (0.0..width as f32).contains(&x) && (0.0..height as f32).contains(&y),
            "probe {} at {:?} lies outside the image",
            probe.bit,
            probe.ndc
        );
        let offset = (y as usize * width as usize + x as usize) * 4;
        let pixel = &pixels[offset..offset + 3];
        let set = distance(pixel, probe.on) < distance(pixel, probe.off);
        value = (value << 1) | u32::from(set);
    }
    Ok(value)
}

fn distance(pixel: &[u8], color: [u8; 3]) -> u32 {
    pixel
        .iter()
        .zip(color)
        .map(|(&a, b)| u32::from(a.abs_diff(b)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ON: [u8; 3] = [255, 255, 255];
    const OFF: [u8; 3] = [70, 70, 70];

    fn probes() -> Vec<Probe> {
        (0..4)
            .map(|bit| Probe {
                bit,
                ndc: [-0.75 + bit as f32 * 0.5, 0.0],
                on: ON,
                off: OFF,
            })
            .collect()
    }

    fn image(bits: [bool; 4]) -> Vec<u8> {
        let (width, height) = (8usize, 2usize);
        let mut pixels = vec![0; width * height * 4];
        for (bit, set) in bits.into_iter().enumerate() {
            let x = 1 + bit * 2;
            let offset = (width + x) * 4;
            let color = if set { ON } else { OFF };
            pixels[offset..offset + 3].copy_from_slice(&color);
        }
        pixels
    }

    #[test]
    fn test_decode_counter_reads_msb_first() {
        let pixels = image([true, false, true, true]);
        assert_eq!(
            decode_counter(&pixels, 8, 2, &probes()).expect("decode"),
            0b1011
        );
    }

    #[test]
    fn test_decode_counter_order_independent_of_probe_order() {
        let pixels = image([false, true, false, false]);
        let mut reversed = probes();
        reversed.reverse();
        assert_eq!(
            decode_counter(&pixels, 8, 2, &reversed).expect("decode"),
            0b0100
        );
    }

    #[test]
    fn test_decode_counter_wrong_size_fails() {
        assert!(decode_counter(&[0; 12], 8, 2, &probes()).is_err());
    }
}
