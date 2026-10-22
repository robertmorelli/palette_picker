// color conversions on the CPU: Lab, CIECAM16 perceived color, gradients
const hex = (rgba) => "#" + [rgba & 255, (rgba >> 8) & 255, (rgba >> 16) & 255].map((v) => v.toString(16).padStart(2, "0")).join("");

// sRGB (packed RGBA) -> Lab (D65), same as the shader
function toLab(rgba) {
  const lin = [rgba & 255, (rgba >> 8) & 255, (rgba >> 16) & 255].map((v) => { v /= 255; return v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92; });
  const M = [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]];
  const xyz = M.map((row, i) => row.reduce((s, m, j) => s + m * lin[j], 0) / [0.95047, 1, 1.08883][i]);
  const f = xyz.map((t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116));
  return [116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])];
}

// CIECAM16 with full adaptation (D = 1) and average surround; the response function has no +0.1 offset, so the
// forward and inverse models undo each other exactly
const mul3 = (M, v) => M.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
function invert3([[a, b, c], [d, e, f], [g, h, i]]) {
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A, c * h - b * i, b * f - c * e], [B, a * i - c * g, c * d - a * f], [C, b * g - a * h, a * e - b * d]].map((r) => r.map((x) => x / det));
}
const M16 = [[0.401288, 0.650173, -0.051461], [-0.250268, 1.204414, 0.045854], [-0.002079, 0.048952, 0.953127]];
const M16inv = invert3(M16);
const SRGB_XYZ = [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]];
const XYZ_SRGB = invert3(SRGB_XYZ);
const SRGB_LIN = Float64Array.from({ length: 256 }, (_, v) => { v /= 255; return v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92; });

// viewing conditions: white (XYZ, Y = 100), adapting luminance La (cd/m2), background luminance Yb (% of white)
function camConditions(white, La, Yb) {
  const c = 0.69, Nc = 1;
  const k4 = (1 / (5 * La + 1)) ** 4;
  const Fl = 0.2 * k4 * 5 * La + 0.1 * (1 - k4) ** 2 * Math.cbrt(5 * La);
  const n = Yb / white[1], z = 1.48 + Math.sqrt(n), Nbb = 0.725 * n ** -0.2;
  const rgbW = mul3(M16, white);
  const Dr = rgbW.map((v) => white[1] / v);
  const adapt = (x) => { const p = (Fl * Math.abs(x) / 100) ** 0.42; return Math.sign(x) * 400 * p / (p + 27.13); };
  const aw = rgbW.map((v, i) => adapt(v * Dr[i]));
  return { c, Nc, Fl, n, z, Nbb, Dr, adapt, Aw: (2 * aw[0] + aw[1] + aw[2] / 20) * Nbb };
}

// XYZ -> lightness J, chroma C, hue h (radians)
function camForward(xyz, v) {
  const [r, g, b] = mul3(M16, xyz).map((x, i) => v.adapt(x * v.Dr[i]));
  const a = r - 12 * g / 11 + b / 11, bb = (r + g - 2 * b) / 9, h = Math.atan2(bb, a);
  const J = 100 * Math.max(0, (2 * r + g + b / 20) * v.Nbb / v.Aw) ** (v.c * v.z);
  const t = (50000 / 13 * v.Nc * v.Nbb * 0.25 * (Math.cos(h + 2) + 3.8) * Math.hypot(a, bb)) / (r + g + 21 * b / 20);
  return { J, C: Math.max(0, t) ** 0.9 * Math.sqrt(J / 100) * (1.64 - 0.29 ** v.n) ** 0.73, h };
}

// J, C, h -> XYZ under the given conditions
function camInverse({ J, C, h }, v) {
  const t = J > 0 ? (C / (Math.sqrt(J / 100) * (1.64 - 0.29 ** v.n) ** 0.73)) ** (1 / 0.9) : 0;
  const K = 50000 / 13 * v.Nc * v.Nbb * 0.25 * (Math.cos(h + 2) + 3.8);
  const p2 = v.Aw * (J / 100) ** (1 / (v.c * v.z)) / v.Nbb;
  const resp = (p, a, b) => [(460 * p + 451 * a + 288 * b) / 1403, (460 * p - 891 * a - 261 * b) / 1403, (460 * p - 220 * a - 6300 * b) / 1403];
  const u = ([r, g, b]) => r + g + 21 * b / 20;
  // t * u(responses) = K * gamma, with u linear in (p2, gamma): solve for gamma, the a/b magnitude
  const gamma = t > 0 ? t * u(resp(p2, 0, 0)) / (K - t * u(resp(0, Math.cos(h), Math.sin(h)))) : 0;
  const unadapt = (x) => { const m = Math.min(Math.abs(x), 399.99); return Math.sign(x) * (100 / v.Fl) * (27.13 * m / (400 - m)) ** (1 / 0.42); };
  return mul3(M16inv, resp(p2, gamma * Math.cos(h), gamma * Math.sin(h)).map((x, i) => unadapt(x) / v.Dr[i]));
}

// a packed sRGB color seen under `from` (XYZ scaled by `scale` so the photo's white is Y = 100), re-rendered as the
// color that looks the same under `to`
function perceive(rgba, from, to, scale) {
  const xyz = mul3(SRGB_XYZ, [rgba & 255, (rgba >> 8) & 255, (rgba >> 16) & 255].map((x) => SRGB_LIN[x])).map((x) => x * scale);
  const out = mul3(XYZ_SRGB, camInverse(camForward(xyz, from), to).map((x) => x / 100));
  const [r, g, b] = out.map((x) => { x = Math.min(1, Math.max(0, x)); return Math.round(255 * (x > 0.0031308 ? 1.055 * x ** (1 / 2.4) - 0.055 : 12.92 * x)); });
  return (r | (g << 8) | (b << 16) | 0xff000000) >>> 0;
}

// the photo's white (raw XYZ, 0-1): the brightest 5% of near-neutral, unclipped pixels, which in an eye photo are the
// eye whites; their tint is the light's color. falls back to D65 (no correction) if there are none
function estimateWhite(pixels) {
  const step = Math.max(1, Math.floor(pixels.length / 200000)), found = [];
  for (let i = 0; i < pixels.length; i += step) {
    const p = pixels[i], r = p & 255, g = (p >> 8) & 255, b = (p >> 16) & 255;
    if (Math.max(r, g, b) >= 250) continue;  // clipped highlights carry no color
    const lab = toLab(p);
    if (Math.hypot(lab[1], lab[2]) <= 25) found.push([lab[0], r, g, b]);
  }
  if (!found.length) return [0.95047, 1, 1.08883];
  found.sort((x, y) => y[0] - x[0]);
  const top = found.slice(0, Math.ceil(found.length * 0.05));
  return mul3(SRGB_XYZ, [1, 2, 3].map((j) => top.reduce((s, c) => s + SRGB_LIN[c[j]], 0) / top.length));
}

// Lab -> sRGB hex, the inverse of toLab (clamped to the sRGB gamut)
function labToHex([L, a, b]) {
  const fy = (L + 16) / 116;
  const xyz = [a / 500 + fy, fy, fy - b / 200].map((t, i) => (t > 0.206893 ? t ** 3 : (t - 16 / 116) / 7.787) * [0.95047, 1, 1.08883][i]);
  const Minv = [[3.240479, -1.53715, -0.498535], [-0.969256, 1.875992, 0.041556], [0.055648, -0.204043, 1.057311]];
  return "#" + Minv.map((row) => {
    const v = Math.min(1, Math.max(0, row.reduce((s, m, j) => s + m * xyz[j], 0)));
    return Math.round((v > 0.0031308 ? 1.055 * v ** (1 / 2.4) - 0.055 : 12.92 * v) * 255).toString(16).padStart(2, "0");
  }).join("");
}

// PCA on Lab colors: the top component (power iteration on the 3x3 covariance), as a CSS gradient from the members'
// lowest to highest projection, oriented dark to light; sampled in Lab so the gradient follows the component exactly
function pcaGradient(labs) {
  const mean = [0, 1, 2].map((j) => labs.reduce((s, l) => s + l[j], 0) / labs.length);
  const cov = [0, 1, 2].map((i) => [0, 1, 2].map((j) => labs.reduce((s, l) => s + (l[i] - mean[i]) * (l[j] - mean[j]), 0) / labs.length));
  let v = [1, 1, 1].map((x) => x / Math.sqrt(3));
  for (let it = 0; it < 50; it++) {
    const u = cov.map((row) => row.reduce((s, c, j) => s + c * v[j], 0));
    const norm = Math.hypot(...u);
    if (norm < 1e-9) break;  // one color, or all identical: no spread
    v = u.map((x) => x / norm);
  }
  if (v[0] < 0) v = v.map((x) => -x);  // dark to light
  const ts = labs.map((l) => l.reduce((s, x, j) => s + (x - mean[j]) * v[j], 0));
  const lo = Math.min(...ts), hi = Math.max(...ts);
  const colors = Array.from({ length: 9 }, (_, i) => labToHex(mean.map((m, j) => m + (lo + ((hi - lo) * i) / 8) * v[j])));
  return { css: `linear-gradient(to right, ${colors.map((c, i) => `${c} ${i * 12.5}%`).join(", ")})`, lo: colors[0], hi: colors[8] };
}

export { hex, toLab, SRGB_LIN, camConditions, perceive, estimateWhite, pcaGradient };
