//! Pixel measurements on decoded RGBA8 captures.
//!
//! Positions are in pixel units with pixel centres at `i + 0.5`, so a quad covering
//! pixels 10..=19 has its centroid at 15.0 and NDC 0 maps to `width / 2`.

use s3d_shared::capture::{Image, Sidecar};

pub type Rgb = [u8; 3];

/// Colour of pixel (`x`, `y`).
pub fn pixel(image: &Image, x: u32, y: u32) -> Rgb {
    let offset = (y as usize * image.width as usize + x as usize) * 4;
    [
        image.rgba[offset],
        image.rgba[offset + 1],
        image.rgba[offset + 2],
    ]
}

/// Whether every channel differs by at most `tolerance`.
pub fn matches(a: Rgb, b: Rgb, tolerance: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tolerance)
}

/// Pixels of one colour: their count, centroid and bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blob {
    pub count: usize,
    pub centroid: (f64, f64),
    /// Pixel-centre coordinates of the extreme pixels.
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

/// Inclusive-exclusive pixel rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Rect {
    pub fn whole(image: &Image) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: image.width,
            y1: image.height,
        }
    }

    /// Square of half-size `radius` around (`x`, `y`), clipped to the image.
    pub fn around(image: &Image, x: f64, y: f64, radius: f64) -> Self {
        let clamp = |value: f64, limit: u32| value.clamp(0.0, f64::from(limit)) as u32;
        Self {
            x0: clamp(x - radius, image.width),
            y0: clamp(y - radius, image.height),
            x1: clamp(x + radius, image.width),
            y1: clamp(y + radius, image.height),
        }
    }

    /// Every (`x`, `y`) inside, row by row.
    pub(crate) fn pixels(self) -> impl Iterator<Item = (u32, u32)> {
        (self.y0..self.y1).flat_map(move |y| (self.x0..self.x1).map(move |x| (x, y)))
    }
}

/// Every pixel in `region` passing `keep`, summarised; `None` when there is none.
pub fn blob_where(image: &Image, region: Rect, keep: impl Fn(Rgb) -> bool) -> Option<Blob> {
    let mut count = 0usize;
    let (mut sum_x, mut sum_y) = (0.0, 0.0);
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for (x, y) in region.pixels() {
        if !keep(pixel(image, x, y)) {
            continue;
        }
        let (cx, cy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
        count += 1;
        sum_x += cx;
        sum_y += cy;
        min_x = min_x.min(cx);
        max_x = max_x.max(cx);
        min_y = min_y.min(cy);
        max_y = max_y.max(cy);
    }
    (count > 0).then(|| Blob {
        count,
        centroid: (sum_x / count as f64, sum_y / count as f64),
        min_x,
        max_x,
        min_y,
        max_y,
    })
}

/// Pixels within `tolerance` of `color`.
pub fn find_color(image: &Image, color: Rgb, tolerance: u8) -> Option<Blob> {
    blob_where(image, Rect::whole(image), |p| matches(p, color, tolerance))
}

/// Fraction of all pixels within `tolerance` of `color`.
pub fn fraction_of(image: &Image, color: Rgb, tolerance: u8) -> f64 {
    let total = f64::from(image.width) * f64::from(image.height);
    let count = blob_where(image, Rect::whole(image), |p| matches(p, color, tolerance))
        .map_or(0, |blob| blob.count);
    count as f64 / total
}

/// Rec. 601 luma in [0, 255].
pub fn luma(color: Rgb) -> f64 {
    0.299 * f64::from(color[0]) + 0.587 * f64::from(color[1]) + 0.114 * f64::from(color[2])
}

/// `color` alpha-blended over `background` as an 8-bit UNORM target stores it.
pub fn blend(color: Rgb, alpha: f32, background: Rgb) -> Rgb {
    let mut out = [0; 3];
    for (channel, (c, b)) in out.iter_mut().zip(color.into_iter().zip(background)) {
        let value = f32::from(c) / 255.0 * alpha + f32::from(b) / 255.0 * (1.0 - alpha);
        *channel = (value * 255.0).round() as u8;
    }
    out
}

/// Vulkan NDC of a world point through the sidecar's camera.
pub fn project(sidecar: &Sidecar, world: [f32; 3]) -> [f32; 3] {
    let view = transform(&sidecar.view, [world[0], world[1], world[2], 1.0]);
    let clip = transform(&sidecar.projection, view);
    [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
}

/// Pixel position (centres at `i + 0.5`) of a Vulkan NDC point.
pub fn ndc_to_pixel(ndc: [f32; 2], width: u32, height: u32) -> (f64, f64) {
    (
        (f64::from(ndc[0]) * 0.5 + 0.5) * f64::from(width),
        (f64::from(ndc[1]) * 0.5 + 0.5) * f64::from(height),
    )
}

/// Column-major 4x4 matrix times a column vector.
fn transform(matrix: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (row, value) in out.iter_mut().enumerate() {
        *value = (0..4).map(|col| matrix[col * 4 + row] * v[col]).sum();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, fill: Rgb) -> Image {
        let rgba = (0..width * height)
            .flat_map(|_| [fill[0], fill[1], fill[2], 255])
            .collect();
        Image {
            width,
            height,
            rgba,
        }
    }

    fn paint(image: &mut Image, region: Rect, color: Rgb) {
        for (x, y) in region.pixels() {
            let offset = (y as usize * image.width as usize + x as usize) * 4;
            image.rgba[offset..offset + 3].copy_from_slice(&color);
        }
    }

    #[test]
    fn test_blob_centroid_uses_pixel_centres() {
        let mut img = image(32, 8, [0; 3]);
        let rect = Rect {
            x0: 10,
            y0: 2,
            x1: 20,
            y1: 4,
        };
        paint(&mut img, rect, [9, 9, 9]);
        let blob = find_color(&img, [9, 9, 9], 0).expect("found");
        assert_eq!(blob.count, 20);
        assert_eq!(blob.centroid, (15.0, 3.0));
        assert_eq!((blob.min_x, blob.max_x), (10.5, 19.5));
    }

    #[test]
    fn test_blend_matches_unorm_rounding() {
        assert_eq!(blend([230, 60, 60], 0.5, [40, 40, 40]), [135, 50, 50]);
        assert_eq!(blend([255, 0, 0], 1.0, [9, 9, 9]), [255, 0, 0]);
    }

    #[test]
    fn test_ndc_to_pixel_maps_centre_and_edges() {
        assert_eq!(ndc_to_pixel([0.0, 0.0], 320, 180), (160.0, 90.0));
        assert_eq!(ndc_to_pixel([-1.0, 1.0], 320, 180), (0.0, 180.0));
    }

    #[test]
    fn test_fraction_counts_matching_pixels() {
        let mut img = image(10, 10, [1, 2, 3]);
        paint(
            &mut img,
            Rect {
                x0: 0,
                y0: 0,
                x1: 5,
                y1: 2,
            },
            [200; 3],
        );
        assert!((fraction_of(&img, [1, 2, 3], 0) - 0.9).abs() < 1e-12);
    }
}
