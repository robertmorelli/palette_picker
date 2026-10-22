// WGSL compute shaders and the sizes they are built with
export const WG = 256, PER_THREAD = 64, CHUNK = WG * PER_THREAD, FEATURES = 13, STRIDE = 14;  // 13 features + 1 spare slot per pixel (sums keep the count there)

export const COMMON = /* wgsl */ `
struct Params { n: u32, w: u32, h: u32, k: u32, dark: f32, light: f32, chunks: u32, m: u32, hilN: u32, levels: u32, p0: u32, p1: u32,
                 weights: array<vec4f, 4> };  // weights: L, C, h, brown | white, x, y, Lab L | Lab a, Lab b, Hilbert, Z-order | Peano, -, -, -

// sRGB (packed RGBA u32) -> Lab (D65), matching skimage's rgb2lab
fn lab(c: u32) -> vec3f {
  var rgb = vec3f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
  rgb = select(rgb / 12.92, pow((rgb + 0.055) / 1.055, vec3f(2.4)), rgb > vec3f(0.04045));
  let xyz = vec3f(
    0.412453 * rgb.x + 0.357580 * rgb.y + 0.180423 * rgb.z,
    0.212671 * rgb.x + 0.715160 * rgb.y + 0.072169 * rgb.z,
    0.019334 * rgb.x + 0.119193 * rgb.y + 0.950227 * rgb.z) / vec3f(0.95047, 1.0, 1.08883);
  let f = select(7.787 * xyz + 16.0 / 116.0, pow(xyz, vec3f(1.0 / 3.0)), xyz > vec3f(0.008856));
  return vec3f(116.0 * f.y - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
}

fn flat(wid: vec3u, nw: vec3u, li: u32) -> u32 { return (wid.y * nw.x + wid.x) * ${WG}u + li; }
`;

export const SHADERS = {
  // lightness histogram (0.1 steps over 0-100), for the per-image 5% / 95% rescale
  histogram: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> px: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> P: Params;
var<workgroup> local: array<atomic<u32>, 1001>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                                      @builtin(local_invocation_index) li: u32) {
  let i = flat(wid, nw, li);
  if (i < P.n) { atomicAdd(&local[u32(clamp(round(lab(px[i]).x * 10.0), 0.0, 1000.0))], 1u); }
  workgroupBarrier();
  for (var b = li; b < 1001u; b += ${WG}u) { let v = atomicLoad(&local[b]); if (v > 0u) { atomicAdd(&hist[b], v); } }
}`,

  // 3x3 sharpen (center 5, edge neighbors -1), clamped at the image border; clustering runs on the sharpened pixels
  sharpen: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> px: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> P: Params;
fn rgb(c: u32) -> vec3f { return vec3f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)); }
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                                      @builtin(local_invocation_index) li: u32) {
  let i = flat(wid, nw, li);
  if (i >= P.n) { return; }
  let x = i % P.w; let y = i / P.w;
  let l = select(i - 1u, i, x == 0u); let r = select(i + 1u, i, x + 1u >= P.w);
  let u = select(i - P.w, i, y == 0u); let d = select(i + P.w, i, y + 1u >= P.h);
  let s = clamp(5.0 * rgb(px[i]) - rgb(px[l]) - rgb(px[r]) - rgb(px[u]) - rgb(px[d]), vec3f(0.0), vec3f(255.0));
  let q = vec3u(round(s));
  out[i] = q.x | (q.y << 8u) | (q.z << 16u) | 0xff000000u;
}`,

  // the clustering values per pixel
  features: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> px: array<u32>;
@group(0) @binding(1) var<storage, read_write> feats: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;
fn cosPow(c: f32, power: f32) -> f32 { return pow(clamp(c, 0.0, 1.0), power) * 100.0; }

// distance along the hilbert curve filling an n x n grid (n a power of 2; 2x2 cells, continuous)
fn hilbert(qx: u32, qy: u32, n: u32) -> u32 {
  var x = qx; var y = qy; var d = 0u; var s = n / 2u;
  loop {
    if (s == 0u) { break; }
    let rx = select(0u, 1u, (x & s) > 0u); let ry = select(0u, 1u, (y & s) > 0u);
    d += s * s * ((3u * rx) ^ ry);
    if (ry == 0u) { if (rx == 1u) { x = n - 1u - x; y = n - 1u - y; } let t = x; x = y; y = t; }
    s /= 2u;
  }
  return d;
}

// z-order (morton): interleave the bits of x and y (jumps at every power-of-2 boundary)
fn spread(v: u32) -> u32 {
  var x = v & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu; x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u; x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

// distance along the peano curve filling a 3^levels grid (3x3 cells, serpentine, continuous)
fn peano(qx: u32, qy: u32, levels: u32) -> u32 {
  var d = 0u; var fx = false; var fy = false; var s = 1u;
  for (var l = 1u; l < levels; l++) { s *= 3u; }
  for (var l = 0u; l < levels; l++) {
    var a = (qx / s) % 3u; var b = (qy / s) % 3u;
    if (fx) { a = 2u - a; }
    if (fy) { b = 2u - b; }
    d = d * 9u + 3u * a + select(b, 2u - b, a % 2u == 1u);  // up column 0, down column 1, up column 2
    fy = fy != (a % 2u == 1u); fx = fx != (b % 2u == 1u);
    s /= 3u;
  }
  return d;
}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                                      @builtin(local_invocation_index) li: u32) {
  let i = flat(wid, nw, li);
  if (i >= P.n) { return; }
  let c = lab(px[i]);
  let L = clamp((c.x - P.dark) / max(P.light - P.dark, 1e-6) * 100.0, 0.0, 100.0);  // relative to this image
  let C = length(c.yz);
  var h = select(atan2(c.z, c.y), 0.0, C < 1e-4);  // atan2(0, 0) is NaN on some GPUs (pure grays)
  if (h < 0.0) { h += 6.283185307; }
  h *= 100.0 / 6.283185307;
  let lch = vec3f(L, C, h);
  // brownness: cosine to a typical skin tone (L 55, C 22, h 48 deg), ^5, scaled by inverse lightness
  let skin = vec3f(55.0, 22.0, 48.0 / 3.6);
  let brown = cosPow(dot(lch, skin) / max(length(lch), 1e-6) / length(skin), 5.0) * clamp(1.0 - L / 100.0, 0.0, 1.0);
  // whiteness: cosine of (L, C) to a light neutral (L 85, C 3), ^20, scaled by lightness so dark grays don't count as white
  let neutral = vec2f(85.0, 3.0);
  let white = cosPow(dot(lch.xy, neutral) / max(length(lch.xy), 1e-6) / length(neutral), 20.0) * clamp(L / 85.0, 0.0, 1.0);
  let s = 100.0 / f32(max(P.w, P.h));
  let a = P.weights[0]; let b = P.weights[1]; let w3 = P.weights[2]; let w4 = P.weights[3];
  let qx = i % P.w; let qy = i / P.w;
  let cells = f32(P.hilN * P.hilN);
  let hil = f32(hilbert(qx, qy, P.hilN)) / cells * 100.0;
  let zo = f32(spread(qx) | (spread(qy) << 1u)) / cells * 100.0;
  let pe = f32(peano(qx, qy, P.levels)) / pow(9.0, f32(P.levels)) * 100.0;
  let o = i * ${STRIDE}u;
  feats[o] = L * a.x; feats[o + 1u] = C * a.y; feats[o + 2u] = h * a.z; feats[o + 3u] = brown * a.w; feats[o + 4u] = white * b.x;
  feats[o + 5u] = f32(i % P.w) * s * b.y; feats[o + 6u] = f32(i / P.w) * s * b.z; feats[o + 7u] = c.x * b.w;
  feats[o + 8u] = c.y * w3.x; feats[o + 9u] = c.z * w3.y; feats[o + 10u] = hil * w3.z; feats[o + 11u] = zo * w3.w;
  feats[o + 12u] = pe * w4.x; feats[o + 13u] = 0.0;
}`,

  // copy a sample of pixels' values out, for k-means++ starting points
  gather: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> feats: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.m) { return; }
  for (var d = 0u; d < ${STRIDE}u; d++) { out[g.x * ${STRIDE}u + d] = feats[idx[g.x] * ${STRIDE}u + d]; }
}`,

  // nearest center for each pixel, and its squared distance (for medoids)
  assign: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> feats: array<f32>;
@group(0) @binding(1) var<storage, read> centers: array<f32>;
@group(0) @binding(2) var<storage, read_write> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> dist: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                                      @builtin(local_invocation_index) li: u32) {
  let i = flat(wid, nw, li);
  if (i >= P.n) { return; }
  var best = 3.4e38; var bestK = 0u;
  for (var c = 0u; c < P.k; c++) {
    var d = 0.0;
    for (var j = 0u; j < ${FEATURES}u; j++) { let t = feats[i * ${STRIDE}u + j] - centers[c * ${STRIDE}u + j]; d += t * t; }
    if (d < best) { best = d; bestK = c; }
  }
  labels[i] = bestK; dist[i] = best;
}`,

  // per (cluster, chunk of pixels): sum of values and count, reduced in workgroup memory (no global float atomics)
  sums: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> feats: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> sh: array<array<f32, ${STRIDE}>, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let chunk = wid.x; let c = wid.y;
  var acc = array<f32, ${STRIDE}>();
  let end = min(P.n, (chunk + 1u) * ${CHUNK}u);
  for (var i = chunk * ${CHUNK}u + li; i < end; i += ${WG}u) {
    if (labels[i] == c) {
      for (var j = 0u; j < ${FEATURES}u; j++) { acc[j] += feats[i * ${STRIDE}u + j]; }
      acc[${STRIDE - 1}] += 1.0;  // count (the last slot is padding in the features)
    }
  }
  sh[li] = acc;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { for (var j = 0u; j < ${STRIDE}u; j++) { sh[li][j] += sh[li + s][j]; } }
    workgroupBarrier();
  }
  if (li == 0u) { for (var j = 0u; j < ${STRIDE}u; j++) { partial[(c * P.chunks + chunk) * ${STRIDE}u + j] = sh[0][j]; } }
}`,

  // per (cluster, chunk): the pixel closest to the cluster's center
  medoid: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> labels: array<u32>;
@group(0) @binding(1) var<storage, read> dist: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<vec2u>;
@group(0) @binding(3) var<uniform> P: Params;
var<workgroup> shD: array<f32, ${WG}>;
var<workgroup> shI: array<u32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let chunk = wid.x; let c = wid.y;
  var best = 3.4e38; var bestI = 0xffffffffu;
  let end = min(P.n, (chunk + 1u) * ${CHUNK}u);
  for (var i = chunk * ${CHUNK}u + li; i < end; i += ${WG}u) {
    if (labels[i] == c && dist[i] < best) { best = dist[i]; bestI = i; }
  }
  shD[li] = best; shI[li] = bestI;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (li < s && shD[li + s] < shD[li]) { shD[li] = shD[li + s]; shI[li] = shI[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { partial[c * P.chunks + chunk] = vec2u(bitcast<u32>(shD[0]), shI[0]); }
}`,

  // paint each pixel its cluster's medoid color
  render: COMMON + /* wgsl */ `
@group(0) @binding(0) var<storage, read> labels: array<u32>;
@group(0) @binding(1) var<storage, read> palette: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                                      @builtin(local_invocation_index) li: u32) {
  let i = flat(wid, nw, li);
  if (i < P.n) { out[i] = palette[labels[i]]; }
}`,
};
