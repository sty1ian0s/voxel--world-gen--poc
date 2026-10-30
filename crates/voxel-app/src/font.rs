//! A 5x7 bitmap font, generated rather than shipped as an asset.
//!
//! The HUD needs to draw digits and a few labels, and nothing else. A font
//! crate plus a `.ttf` file would be a large amount of machinery for thirty-odd
//! glyphs, so the glyphs are encoded here as bit rows: five bits per row, seven
//! rows per glyph, one `u8` per row with the top five bits used.
//!
//! Layout:
//!
//! ```text
//!   0b11111   = a full top row
//!   0b10001   = left and right pixels
//! ```
//!
//! Each glyph is 6 pixels wide on screen (five plus a blank column) and 8 tall
//! (seven plus a blank row), which also gives integer scaling for a larger HUD.

/// Width of a glyph's ink, in pixels.
pub const GLYPH_W: u32 = 5;
/// Height of a glyph's ink, in pixels.
pub const GLYPH_H: u32 = 7;
/// Horizontal advance, ink plus one blank column so characters do not touch.
pub const ADVANCE_X: u32 = GLYPH_W + 1;
/// Vertical advance, ink plus one blank row.
pub const ADVANCE_Y: u32 = GLYPH_H + 1;

/// One glyph as seven rows, MSB-first in the low five bits.
pub type Glyph = [u8; 7];

const fn g(rows: [u8; 7]) -> Glyph {
    rows
}

/// The glyphs the HUD needs: digits, the letters used in labels, and a little
/// punctuation. Anything else renders as `?`, which is deliberate: a missing
/// glyph should be obvious rather than blank.
pub fn glyph_for(c: char) -> Option<Glyph> {
    Some(match c {
        '0' => g([0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
        '1' => g([0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        '2' => g([0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
        '3' => g([0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110]),
        '4' => g([0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
        '5' => g([0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
        '6' => g([0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
        '7' => g([0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
        '8' => g([0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
        '9' => g([0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100]),
        'A' => g([0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
        'B' => g([0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110]),
        'C' => g([0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
        'D' => g([0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
        'E' => g([0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
        'F' => g([0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
        'G' => g([0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111]),
        'H' => g([0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
        'I' => g([0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        'J' => g([0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100]),
        'K' => g([0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
        'L' => g([0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
        'M' => g([0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
        'N' => g([0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
        'O' => g([0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
        'P' => g([0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
        'Q' => g([0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101]),
        'R' => g([0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001]),
        'S' => g([0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110]),
        'T' => g([0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
        'U' => g([0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
        'V' => g([0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
        'W' => g([0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001]),
        'X' => g([0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001]),
        'Y' => g([0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100]),
        'Z' => g([0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111]),
        'a' => g([0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111]),
        'e' => g([0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110]),
        'i' => g([0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110]),
        'm' => g([0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101]),
        'n' => g([0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001]),
        'o' => g([0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110]),
        's' => g([0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110]),
        'p' => g([0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000]),
        'r' => g([0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000]),
        't' => g([0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00100, 0b00011]),
        'v' => g([0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
        'x' => g([0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001]),
        'd' => g([0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111]),
        'g' => g([0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110]),
        'c' => g([0b00000, 0b00000, 0b01110, 0b10001, 0b10000, 0b10001, 0b01110]),
        'l' => g([0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
        'h' => g([0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001]),
        'u' => g([0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101]),
        'f' => g([0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000]),
        'b' => g([0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110]),
        'k' => g([0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010]),
        'j' => g([0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100]),
        'q' => g([0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001]),
        'w' => g([0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010]),
        'y' => g([0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110]),
        'z' => g([0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111]),
        ' ' => g([0; 7]),
        '.' => g([0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100]),
        ',' => g([0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b00100, 0b01000]),
        ':' => g([0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000]),
        ';' => g([0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000]),
        '/' => g([0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000]),
        '-' => g([0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000]),
        '+' => g([0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000]),
        '%' => g([0b11001, 0b11010, 0b00010, 0b00100, 0b01011, 0b10011, 0b00000]),
        '|' => g([0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
        '[' => g([0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110]),
        ']' => g([0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110]),
        '_' => g([0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111]),
        '?' => g([0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100]),
        _ => return None,
    })
}

/// The glyph used when a character has none.
pub const FALLBACK: Glyph = [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100];

/// Pixel size of a block of text at a given integer scale.
pub fn text_size(text: &str, scale: u32) -> (u32, u32) {
    let scale = scale.max(1);
    let chars = text.chars().count() as u32;
    if chars == 0 {
        return (0, 0);
    }
    // Every glyph advances one blank column, so the last one leaves a trailing
    // blank; subtract it so the reported width is ink-accurate.
    let w = (chars * ADVANCE_X).saturating_sub(1);
    (w * scale, GLYPH_H * scale)
}

/// An RGBA pixel buffer holding one line of text, ready to upload as a texture.
///
/// One texture per HUD line keeps each line independently positionable and
/// sized, and the HUD redraws rarely enough that the per-line upload cost does
/// not matter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextTexture {
    pub width: u32,
    pub height: u32,
    /// RGBA8, `width * height * 4` bytes.
    pub pixels: Vec<u8>,
}

/// Renders `text` into an RGBA buffer.
///
/// `fg` is the glyph colour and `bg` the background, both `[r, g, b]`. The
/// background is written into every pixel so the HUD stays readable over bright
/// terrain; passing a zero alpha is not offered because a fully transparent
/// background needs blending, which this HUD deliberately avoids.
pub fn render_text(text: &str, scale: u32, fg: [u8; 3], bg: [u8; 3]) -> TextTexture {
    let scale = scale.max(1);
    let (w, h) = text_size(text, scale);
    if w == 0 || h == 0 {
        return TextTexture {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
    }

    // Start filled with the background, then stamp glyph pixels over it.
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for px in pixels.as_chunks_mut::<4>().0 {
        px[0] = bg[0];
        px[1] = bg[1];
        px[2] = bg[2];
        px[3] = 255;
    }

    for (i, c) in text.chars().enumerate() {
        let glyph = glyph_for(c).unwrap_or(FALLBACK);
        let origin_x = i as u32 * ADVANCE_X * scale;

        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..GLYPH_W {
                // Bit 4 is the leftmost pixel of the five.
                let on = bits & (1 << (GLYPH_W - 1 - col)) != 0;
                if !on {
                    continue;
                }
                // Scale the pixel into a `scale x scale` block.
                for sy in 0..scale {
                    for sx in 0..scale {
                        let x = origin_x + col * scale + sx;
                        let y = row as u32 * scale + sy;
                        if x >= w || y >= h {
                            continue;
                        }
                        let idx = ((y * w + x) * 4) as usize;
                        pixels[idx] = fg[0];
                        pixels[idx + 1] = fg[1];
                        pixels[idx + 2] = fg[2];
                        pixels[idx + 3] = 255;
                    }
                }
            }
        }
    }

    TextTexture {
        width: w,
        height: h,
        pixels,
    }
}

/// Renders `text` into a buffer of a caller-chosen width, padding with the
/// background colour.
///
/// The HUD draws one panel behind all of its lines, so every line needs to be
/// the same width. Rendering each at its natural width and letting the panel
/// show through gives a ragged right edge, which reads as a rendering bug
/// rather than as a design.
pub fn render_text_padded(
    text: &str,
    scale: u32,
    fg: [u8; 3],
    bg: [u8; 3],
    width: u32,
) -> TextTexture {
    let inner = render_text(text, scale, fg, bg);
    let height = inner.height;
    if width == 0 || height == 0 {
        return inner;
    }
    if inner.width >= width {
        // Too long to fit: the caller decides whether to clip or widen. The
        // text is returned unchanged so nothing is silently lost.
        return inner;
    }

    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for px in pixels.as_chunks_mut::<4>().0 {
        px[0] = bg[0];
        px[1] = bg[1];
        px[2] = bg[2];
        px[3] = 255;
    }
    // Copy the rendered line into the left of the wider buffer.
    for y in 0..height {
        let src = (y * inner.width * 4) as usize;
        let dst = (y * width * 4) as usize;
        let len = (inner.width * 4) as usize;
        pixels[dst..dst + len].copy_from_slice(&inner.pixels[src..src + len]);
    }

    TextTexture {
        width,
        height,
        pixels,
    }
}

/// Background colour for the HUD panels: near-black with a little warmth so it
/// reads as an overlay rather than as part of the world.
pub const HUD_BG: [u8; 3] = [12, 14, 18];
/// Text colour, near-white.
pub const HUD_FG: [u8; 3] = [226, 232, 240];
/// Accent used for the highlighted value in a line.
pub const HUD_ACCENT: [u8; 3] = [120, 220, 160];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_digit_and_letter_has_a_glyph() {
        for c in '0'..='9' {
            assert!(glyph_for(c).is_some(), "missing digit {c}");
        }
        for c in 'A'..='Z' {
            assert!(glyph_for(c).is_some(), "missing uppercase {c}");
        }
        for c in 'a'..='z' {
            assert!(glyph_for(c).is_some(), "missing lowercase {c}");
        }
    }

    #[test]
    fn unknown_characters_fall_back_rather_than_vanish() {
        assert!(glyph_for('~').is_none(), "~ should have no glyph");
        let t = render_text("~", 1, HUD_FG, HUD_BG);
        // The fallback is not blank, so the character is visibly wrong on
        // screen rather than silently missing.
        let ink = t
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] == HUD_FG[0] && p[1] == HUD_FG[1])
            .count();
        assert!(ink > 0, "fallback glyph drew nothing");
    }

    #[test]
    fn glyphs_use_only_the_low_five_bits() {
        // A stray high bit would bleed into the neighbouring character's column.
        for c in "0123456789ABCXYZabcxyz .:-/%?[]".chars() {
            let gl = glyph_for(c).unwrap();
            for (row, bits) in gl.iter().enumerate() {
                assert_eq!(
                    bits & 0b1110_0000,
                    0,
                    "glyph {c:?} row {row} sets a bit above the fifth column"
                );
            }
        }
    }

    #[test]
    fn no_glyph_is_completely_blank_except_space() {
        for c in "0123456789ABCXYZabcxyz.:-/%;+?[]".chars() {
            let gl = glyph_for(c).unwrap();
            assert!(
                gl.iter().any(|&r| r != 0),
                "glyph {c:?} is blank but is not a space"
            );
        }
        assert!(glyph_for(' ').unwrap().iter().all(|&r| r == 0));
    }

    #[test]
    fn text_size_accounts_for_the_trailing_advance() {
        // One character is 5 wide, not 6: the blank column after the last glyph
        // must not be counted.
        assert_eq!(text_size("A", 1), (5, 7));
        assert_eq!(text_size("AB", 1), (11, 7));
        assert_eq!(text_size("ABC", 1), (17, 7));
        assert_eq!(text_size("", 1), (0, 0));
        assert_eq!(text_size("A", 2), (10, 14));
        assert_eq!(text_size("A", 0), (5, 7), "scale 0 must clamp to 1");
    }

    #[test]
    fn rendering_produces_a_tightly_packed_buffer() {
        let t = render_text("60 fps", 1, HUD_FG, HUD_BG);
        let (w, h) = text_size("60 fps", 1);
        assert_eq!(t.width, w);
        assert_eq!(t.height, h);
        assert_eq!(t.pixels.len(), (w * h * 4) as usize);
    }

    #[test]
    fn background_fills_every_pixel_and_ink_is_present() {
        let t = render_text("8", 1, HUD_FG, HUD_BG);
        // Every pixel is opaque, so the HUD needs no blending to look right.
        assert!(
            t.pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
            "background must be opaque"
        );
        let ink = t
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| [p[0], p[1], p[2]] == HUD_FG)
            .count();
        assert!(ink > 0, "no foreground pixels were drawn");
        let bg = t
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| [p[0], p[1], p[2]] == HUD_BG)
            .count();
        assert!(bg > 0, "background was entirely covered");
    }

    #[test]
    fn scaling_multiplies_the_ink_area_by_the_square() {
        let small = render_text("8", 1, HUD_FG, HUD_BG);
        let big = render_text("8", 2, HUD_FG, HUD_BG);
        let ink = |t: &TextTexture| {
            t.pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| [p[0], p[1], p[2]] == HUD_FG)
                .count()
        };
        assert_eq!(big.width, small.width * 2);
        assert_eq!(big.height, small.height * 2);
        assert_eq!(
            ink(&big),
            ink(&small) * 4,
            "scaling by 2 must quadruple the ink area"
        );
    }

    #[test]
    fn characters_do_not_overlap() {
        // Two glyphs must leave the blank advance column between them. Rendering
        // "II" and checking the middle column is background catches a wrong
        // advance, which would smear text together.
        let t = render_text("II", 1, HUD_FG, HUD_BG);
        let col = GLYPH_W; // the blank column after the first glyph
        let mut any_ink = false;
        for y in 0..t.height {
            let idx = ((y * t.width + col) * 4) as usize;
            if [t.pixels[idx], t.pixels[idx + 1], t.pixels[idx + 2]] == HUD_FG {
                any_ink = true;
            }
        }
        assert!(!any_ink, "advance column contains ink; glyphs would collide");
    }

    #[test]
    fn a_realistic_hud_line_renders_every_character() {
        let line = "fps 60 | p99 12.34ms";
        let t = render_text(line, 1, HUD_FG, HUD_BG);
        assert!(t.width > 0 && t.height == GLYPH_H);

        // Every non-space character must contribute ink somewhere in its cell.
        for (i, c) in line.chars().enumerate() {
            if c == ' ' {
                continue;
            }
            let x0 = i as u32 * ADVANCE_X;
            let mut ink = false;
            for y in 0..t.height {
                for x in x0..(x0 + GLYPH_W).min(t.width) {
                    let idx = ((y * t.width + x) * 4) as usize;
                    if [t.pixels[idx], t.pixels[idx + 1], t.pixels[idx + 2]] == HUD_FG {
                        ink = true;
                    }
                }
            }
            assert!(ink, "character {c:?} at index {i} drew nothing");
        }
    }

    #[test]
    fn empty_text_is_an_empty_texture() {
        let t = render_text("", 2, HUD_FG, HUD_BG);
        assert_eq!((t.width, t.height), (0, 0));
        assert!(t.pixels.is_empty());
    }

    #[test]
    fn glyphs_have_distinct_shapes() {
        // Guards against a copy-paste error leaving two characters identical,
        // which would make a HUD read as gibberish that still looks plausible.
        let mut seen: Vec<(char, Glyph)> = Vec::new();
        for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars() {
            let gl = glyph_for(c).unwrap();
            for (other, other_glyph) in &seen {
                assert_ne!(&gl, other_glyph, "glyph {c:?} is identical to {other:?}");
            }
            seen.push((c, gl));
        }
    }
}
