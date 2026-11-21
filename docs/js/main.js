import { WG, CHUNK, FEATURES, STRIDE, SHADERS } from "./shaders.js";
import { hex, toLab, SRGB_LIN, camConditions, perceive, estimateWhite, pcaGradient } from "./color.js";
// palette extraction on WebGPU, after Comaniciu & Meer (2002): mean shift filtering in the joint (x, y, Lab) domain,
// fusing the modes into regions, and one palette candidate per group of same-colored regions; each is painted its
// medoid (the real pixel closest to its mean). the per-pixel work runs in compute shaders; the CPU fuses regions on a
// downscaled working image and handles small summaries (lightness histogram, per-chunk cluster sums).
// WORK: long side of the working image mean shift runs on; MAX_MODES: most palette candidates before merging
const WORK = 1024, MAX_MODES = 128;
// how much each clustering value counts; tunable from the Weights panel in debug mode
const WEIGHTS = [["LCh L", 0], ["LCh C", 0], ["LCh h", 0], ["Brownness", 0], ["Whiteness", 0], ["x", 0], ["y", 0],
  ["Lab L", 1], ["Lab a", 1], ["Lab b", 1], ["Hilbert", 0], ["Z-order", 0], ["Peano", 0]];

const $ = (id) => document.getElementById(id);
// status text, plus a small square that fills bottom to top with progress (0-1); no progress hides it in place
const status = (text, p) => {
  $("status").textContent = text;
  $("progress").style.visibility = p === undefined ? "hidden" : "";
  $("progress").style.setProperty("--p", p ?? 0);
};


let device, pipelines;

async function initGPU() {
  if (!navigator.gpu) throw new Error("WebGPU isn't available in this browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found.");
  device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.addEventListener("uncapturederror", (e) => status(`GPU error: ${e.error.message}`));
  device.lost.then((info) => { status(`GPU device lost: ${info.message}`); device = null; });
  pipelines = {};
  for (const [name, code] of Object.entries(SHADERS)) {
    const module = device.createShaderModule({ code });
    const errors = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error(`${name} shader: ${errors.map((m) => `line ${m.lineNum}: ${m.message}`).join("; ")}`);
    pipelines[name] = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  }
}

const S = GPUBufferUsage;
const buffer = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
const bind = (name, buffers) => device.createBindGroup({
  layout: pipelines[name].getBindGroupLayout(0),
  entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
});

function run(name, group, x, y = 1) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipelines[name]); pass.setBindGroup(0, group); pass.dispatchWorkgroups(x, y);
  pass.end();
  device.queue.submit([enc.finish()]);
}

// one workgroup per 256 pixels, spread over 2 dimensions to stay under the 65535-per-dimension limit
const pixelGroups = (n) => { const g = Math.ceil(n / WG); const x = Math.min(g, 65535); return [x, Math.ceil(g / x)]; };

async function read(src, size, Type) {
  const staging = buffer(size, S.MAP_READ | S.COPY_DST);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Type(staging.getMappedRange().slice(0));
  staging.destroy();
  return data;
}

function rng(seed) {  // mulberry32: small seeded PRNG for repeatable sampling
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// auto color count: on the sample, keep splitting the widest cluster until every cluster fits within the max spread.
// spread is measured in Lab (95th percentile distance to the cluster's mean, so a few glare pixels don't count), and
// splits are 2-means in the clustering values, so the clusters match what the full run will find
function spreadK(sample, labs, m, radius) {
  const dist = (p, q, n) => { let s = 0; for (let j = 0; j < n; j++) s += (p[j] - q[j]) ** 2; return Math.sqrt(s); };
  const row = (i) => sample.subarray(i * STRIDE, i * STRIDE + FEATURES);
  const centroid = (g) => { const c = new Float64Array(FEATURES); for (const i of g) for (let j = 0; j < FEATURES; j++) c[j] += sample[i * STRIDE + j]; return c.map((v) => v / g.length); };
  const spread = (g) => {
    const mean = [0, 1, 2].map((j) => g.reduce((s, i) => s + labs[i][j], 0) / g.length);
    const d = g.map((i) => dist(labs[i], mean, 3)).sort((a, b) => a - b);
    return d[Math.floor(0.95 * (d.length - 1))];
  };
  const split = (g) => {  // 2-means, started from two far-apart members
    const far = (from) => g.reduce((best, i) => (dist(row(i), row(from), FEATURES) > dist(row(best), row(from), FEATURES) ? i : best), g[0]);
    const ia = far(g[0]), ib = far(ia);
    let a = row(ia).slice(), b = row(ib).slice(), A = [], B = [];
    for (let it = 0; it < 10; it++) {
      A = []; B = [];
      for (const i of g) (dist(row(i), a, FEATURES) <= dist(row(i), b, FEATURES) ? A : B).push(i);
      if (!A.length || !B.length) break;
      a = centroid(A); b = centroid(B);
    }
    return [A, B];
  };
  const clusters = [Array.from({ length: m }, (_, i) => i)];
  while (clusters.length < 64) {
    const spreads = clusters.map(spread);
    const worst = spreads.indexOf(Math.max(...spreads));
    if (spreads[worst] <= radius) break;
    const [A, B] = split(clusters[worst]);
    if (!A.length || !B.length) break;  // can't be split in the clustering values
    clusters.splice(worst, 1, A, B);
  }
  return Math.max(2, clusters.length);
}

// Comaniciu–Meer fusion, on the filtered (mode) colors of the working image, packed as L, a, b, - per pixel:
// 1. neighboring pixels whose modes are within hr join one region
// 2. regions under minSize pixels are folded into their closest-colored neighbor (a few rounds)
// 3. regions anywhere in the image whose mean colors are within hr become one palette candidate, biggest regions
//    first, at most MAX_MODES of them
// returns the candidate of every working pixel, and how many candidates there are
function segment(lab, ww, wh, hr, minSize) {
  const N = ww * wh, hr2 = hr * hr;
  const uf = (size) => {
    const parent = Int32Array.from({ length: size }, (_, i) => i);
    const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    return { find, union: (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; } };
  };
  // regions grow from a seed pixel by flood fill; a neighbor joins when its mode is within hr/2 of the region's running
  // mean (not just of the neighbor it touches), so smooth gradients can't chain far-apart colors into one region
  const region = new Int32Array(N).fill(-1);
  let R = 0, size, sum;
  {
    const queue = new Int32Array(N), half2 = hr2 / 4;
    for (let s = 0; s < N; s++) {
      if (region[s] >= 0) continue;
      let head = 0, tail = 0, count = 1, mL = lab[4 * s], mA = lab[4 * s + 1], mB = lab[4 * s + 2];
      region[s] = R; queue[tail++] = s;
      while (head < tail) {
        const i = queue[head++], x = i % ww;
        for (const j of [x > 0 ? i - 1 : -1, x + 1 < ww ? i + 1 : -1, i - ww, i + ww < N ? i + ww : -1]) {
          if (j < 0 || region[j] >= 0) continue;
          const L = lab[4 * j], A = lab[4 * j + 1], B = lab[4 * j + 2];
          if ((L - mL / count) ** 2 + (A - mA / count) ** 2 + (B - mB / count) ** 2 >= half2) continue;
          region[j] = R; queue[tail++] = j; count++; mL += L; mA += A; mB += B;
        }
      }
      R++;
    }
  }
  const stats = () => {
    size = new Float64Array(R); sum = new Float64Array(R * 3);
    for (let i = 0; i < N; i++) { const r = region[i]; size[r]++; for (let j = 0; j < 3; j++) sum[3 * r + j] += lab[4 * i + j]; }
  };
  const dist2 = (a, b) => { let s = 0; for (let j = 0; j < 3; j++) s += (sum[3 * a + j] / size[a] - sum[3 * b + j] / size[b]) ** 2; return s; };
  stats();
  for (let round = 0; round < 4; round++) {
    const best = new Int32Array(R).fill(-1), bestD = new Float64Array(R).fill(Infinity);
    const consider = (a, b) => { if (a === b || size[a] >= minSize) return; const d = dist2(a, b); if (d < bestD[a]) { bestD[a] = d; best[a] = b; } };
    for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
      const i = y * ww + x;
      if (x + 1 < ww) { consider(region[i], region[i + 1]); consider(region[i + 1], region[i]); }
      if (y + 1 < wh) { consider(region[i], region[i + ww]); consider(region[i + ww], region[i]); }
    }
    if (best.every((b) => b < 0)) break;
    const rg = uf(R);
    for (let a = 0; a < R; a++) if (best[a] >= 0) rg.union(best[a], a);
    const id = new Int32Array(R).fill(-1);
    let next = 0;
    for (let a = 0; a < R; a++) { const r = rg.find(a); if (id[r] < 0) id[r] = next++; }
    for (let i = 0; i < N; i++) region[i] = id[rg.find(region[i])];
    R = next;
    stats();
  }
  // palette candidates: biggest regions first; each joins the closest candidate within hr, or starts a new one
  const order = Array.from({ length: R }, (_, r) => r).sort((a, b) => size[b] - size[a]);
  const cand = new Int32Array(R), cSize = [], cSum = [];
  for (const r of order) {
    const m = [0, 1, 2].map((j) => sum[3 * r + j] / size[r]);
    let bestC = -1, bestD = Infinity;
    for (let c = 0; c < cSize.length; c++) {
      const d = [0, 1, 2].reduce((s, j) => s + (cSum[c][j] / cSize[c] - m[j]) ** 2, 0);
      if (d < bestD) { bestD = d; bestC = c; }
    }
    if (bestC < 0 || (bestD >= hr2 && cSize.length < MAX_MODES)) { bestC = cSize.length; cSize.push(0); cSum.push([0, 0, 0]); }
    cand[r] = bestC; cSize[bestC] += size[r];
    for (let j = 0; j < 3; j++) cSum[bestC][j] += sum[3 * r + j];
  }
  return { labels: Int32Array.from(region, (r) => cand[r]), count: cSize.length, regions: R };
}

async function recolor(bitmap) {
  const t0 = performance.now();
  // number of clusters, from the Colors input
  let K = Math.max(2, Math.min(64, Math.round(Number($("clusters").value)) || 5));
  // if this run takes long enough to notice, hide the previous swatches and lines (in place, so nothing shifts);
  // quick reruns while dragging a slider just swap the new result in, without blinking
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => { current = null; $("chips").style.visibility = $("lines").style.visibility = "hidden"; }, 200);
  const w = bitmap.width, h = bitmap.height, n = w * h;
  const canvas = $("result");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const pixels = new Uint32Array(ctx.getImageData(0, 0, w, h).data.buffer);  // RGBA bytes, little-endian u32

  const chunks = Math.ceil(n / CHUNK);
  const params = new ArrayBuffer(112), pu = new Uint32Array(params), pf = new Float32Array(params);
  // space-filling curve grids: the smallest 2^n (hilbert, z-order) and 3^levels (peano) squares that cover the image
  const hilbertN = 2 ** Math.ceil(Math.log2(Math.max(w, h))), peanoLevels = Math.max(1, Math.ceil(Math.log(Math.max(w, h)) / Math.log(3) - 1e-9));
  const writeParams = (dark, light, k = K) => {
    pu.set([n, w, h, k]); pf.set([dark, light], 4); pu.set([chunks, 0, hilbertN, peanoLevels, 0, 0], 6);
    pf.set(weightValues(), 12);
    device.queue.writeBuffer(paramBuf, 0, params);
  };
  const paramBuf = buffer(112, S.UNIFORM | S.COPY_DST);
  const pxBuf = buffer(n * 4, S.STORAGE | S.COPY_DST);
  device.queue.writeBuffer(pxBuf, 0, pixels);
  const featBuf = buffer(n * STRIDE * 4, S.STORAGE);
  const labelBuf = buffer(n * 4, S.STORAGE | S.COPY_SRC | S.COPY_DST);
  const distBuf = buffer(n * 4, S.STORAGE);
  const [gx, gy] = pixelGroups(n);

  // 1. relative lightness range: the 5th and 95th percentiles of L
  status("Measuring lightness…", 0.05);
  writeParams(0, 100);
  const sharpBuf = buffer(n * 4, S.STORAGE);
  run("sharpen", bind("sharpen", [pxBuf, sharpBuf, paramBuf]), gx, gy);
  const histBuf = buffer(1001 * 4, S.STORAGE | S.COPY_SRC | S.COPY_DST);
  run("histogram", bind("histogram", [sharpBuf, histBuf, paramBuf]), gx, gy);
  const hist = await read(histBuf, 1001 * 4, Uint32Array);
  const percentile = (p) => { let acc = 0; for (let b = 0; b < 1001; b++) { acc += hist[b]; if (acc >= p * n) return b / 10; } return 100; };
  const dark = percentile(0.05), light = percentile(0.95);
  writeParams(dark, light);

  // 2. clustering values, from the original (unsharpened) pixels
  status("Computing features…", 0.1);
  run("features", bind("features", [pxBuf, featBuf, paramBuf]), gx, gy);

  // 3. Comaniciu–Meer mean shift filtering on a working copy of the image (at most WORK pixels on its long side, so the
  // per-pixel windows stay affordable): every pixel climbs to its mode in the joint (x, y, Lab) domain, with a square
  // spatial window of radius hs and a color radius hr in Lab
  const scale = Math.min(1, WORK / Math.max(w, h));
  const ww = Math.max(1, Math.round(w * scale)), wh = Math.max(1, Math.round(h * scale)), N = ww * wh;
  const work = new OffscreenCanvas(ww, wh).getContext("2d");
  work.imageSmoothingQuality = "high";
  work.drawImage(bitmap, 0, 0, ww, wh);
  const workPx = new Uint32Array(work.getImageData(0, 0, ww, wh).data.buffer);
  const hs = Math.round(Number($("spatial-radius").value)), hr = Number($("color-radius").value);
  const fParams = new ArrayBuffer(16);
  new Uint32Array(fParams).set([ww, wh]); new Int32Array(fParams)[2] = hs; new Float32Array(fParams)[3] = hr;
  const fBuf = buffer(16, S.UNIFORM | S.COPY_DST);
  device.queue.writeBuffer(fBuf, 0, fParams);
  const workBuf = buffer(N * 4, S.STORAGE | S.COPY_DST);
  device.queue.writeBuffer(workBuf, 0, workPx);
  const workLabBuf = buffer(N * 16, S.STORAGE);
  const filteredBuf = buffer(N * 16, S.STORAGE | S.COPY_SRC);
  const [fx, fy] = pixelGroups(N);
  status("Mean shift filtering…", 0.2);
  run("labs", bind("labs", [workBuf, workLabBuf, fBuf]), fx, fy);
  run("msfilter", bind("msfilter", [workLabBuf, filteredBuf, fBuf]), fx, fy);
  const filtered = await read(filteredBuf, N * 16, Float32Array);

  // 4. fuse the modes into regions, and the regions into palette candidates (on the CPU; the working image is small)
  status("Fusing regions…", 0.6);
  const seg = segment(filtered, ww, wh, hr, Math.max(20, 0.0002 * N));

  // every full-size pixel takes the candidate of the working pixel it falls in
  const colOf = Int32Array.from({ length: w }, (_, x) => Math.min(ww - 1, Math.floor((x * ww) / w)));
  const labels = new Uint32Array(n);
  for (let y = 0; y < h; y++) {
    const row = Math.min(wh - 1, Math.floor((y * wh) / h)) * ww, o = y * w;
    for (let x = 0; x < w; x++) labels[o + x] = seg.labels[row + colOf[x]];
  }

  // then merge the candidates down (below): auto stops at the min difference, a manual count stops at that count
  const auto = $("clusters-auto").checked;
  const target = auto ? 2 : K;
  K = seg.count;
  writeParams(dark, light);
  const centerBuf = buffer(K * STRIDE * 4, S.STORAGE | S.COPY_DST);
  const sumBuf = buffer(K * chunks * STRIDE * 4, S.STORAGE | S.COPY_SRC);
  const medBuf = buffer(K * chunks * 8, S.STORAGE | S.COPY_SRC);

  // each candidate's center is the mean clustering values of its pixels (summed per (cluster, chunk) on the GPU), and
  // each pixel's distance to its own center picks the medoids
  status("Measuring regions…", 0.85);
  device.queue.writeBuffer(labelBuf, 0, labels);
  run("sums", bind("sums", [featBuf, labelBuf, sumBuf, paramBuf]), chunks, K);
  const partial = await read(sumBuf, K * chunks * STRIDE * 4, Float32Array);
  const clusterSizes = new Float64Array(K);  // pixels per cluster
  const centers = new Float32Array(K * STRIDE);
  for (let c = 0; c < K; c++) {
    const total = new Float64Array(STRIDE);
    for (let ch = 0; ch < chunks; ch++) for (let j = 0; j < STRIDE; j++) total[j] += partial[(c * chunks + ch) * STRIDE + j];
    clusterSizes[c] = total[STRIDE - 1];
    if (total[STRIDE - 1] > 0) for (let j = 0; j < FEATURES; j++) centers[c * STRIDE + j] = total[j] / total[STRIDE - 1];
  }
  device.queue.writeBuffer(centerBuf, 0, centers);
  run("distance", bind("distance", [featBuf, centerBuf, labelBuf, distBuf, paramBuf]), gx, gy);

  // 4. medoids: the real pixel closest to each cluster's center
  status("Finding medoids…", 0.92);
  run("medoid", bind("medoid", [labelBuf, distBuf, medBuf, paramBuf]), chunks, K);
  const medPartial = await read(medBuf, K * chunks * 8, Uint32Array);
  const medPartialF = new Float32Array(medPartial.buffer);
  const medoids = [];
  for (let c = 0; c < K; c++) {
    let best = Infinity, bestI = -1;
    for (let ch = 0; ch < chunks; ch++) {
      const d = medPartialF[(c * chunks + ch) * 2], i = medPartial[(c * chunks + ch) * 2 + 1];
      if (i !== 0xffffffff && d < best) { best = d; bestI = i; }
    }
    medoids.push(bestI);
  }

  // merge the two most similar clusters (by their medoid colors in Lab) until the target count, or for auto until the
  // closest pair is at least the min difference apart; clusters under 0.02% of the pixels (the same floor as the
  // regions, so small but coherent accents survive) are folded into their nearest first, so specks don't. groupOf maps
  // each cluster to the cluster whose color it now shows (the bigger one)
  let groupOf = medoids.map((_, c) => c);
  {
    const labOf = medoids.map((i) => (i < 0 ? null : toLab(pixels[i])));
    const alive = new Set(medoids.flatMap((i, c) => (i < 0 ? [] : [c])));
    const size = Array.from(clusterSizes);
    const de = (a, b) => Math.hypot(labOf[a][0] - labOf[b][0], labOf[a][1] - labOf[b][1], labOf[a][2] - labOf[b][2]);
    const merge = (from, into) => { size[into] += size[from]; alive.delete(from); groupOf = groupOf.map((g) => (g === from ? into : g)); };
    const nearest = (c) => [...alive].filter((d) => d !== c).reduce((best, d) => (best < 0 || de(c, d) < de(c, best) ? d : best), -1);
    for (const c of [...alive]) if (alive.size > target && size[c] < 0.0002 * n) merge(c, nearest(c));  // never below the target
    const minDiff = Number($("min-diff").value);
    while (alive.size > target) {
      let best = Infinity, pa = -1, pb = -1;
      for (const a of alive) for (const b of alive) if (a < b && de(a, b) < best) { best = de(a, b); pa = a; pb = b; }
      if (auto && best >= minDiff) break;
      if (size[pa] >= size[pb]) merge(pb, pa); else merge(pa, pb);
    }
    if (auto) $("clusters").value = alive.size;
  }
  const shown = [...new Set(groupOf)];

  // perceived color (CIECAM16): each color as it looks in the photo (adapted to the photo's white, against the
  // brightness around its region), re-rendered as the color that looks the same on a neutral gray page under D65
  const display = medoids.map((i) => (i < 0 ? 0 : pixels[i]));
  if ($("perceived").checked) {
    const labels = await read(labelBuf, n * 4, Uint32Array);
    const white = estimateWhite(pixels), scale = 100 / white[1];
    // background: the average luminance a short distance outside each color's region (2% of the image size)
    const r = Math.max(2, Math.round(0.02 * Math.max(w, h))), step = Math.max(1, Math.floor(n / 1e6));
    const bgSum = new Float64Array(K), bgN = new Float64Array(K);
    for (let i = 0; i < n; i += step) {
      const x = i % w, y = Math.floor(i / w), g = groupOf[labels[i]];
      for (const [dx, dy] of [[r, 0], [-r, 0], [0, r], [0, -r]]) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx, p = pixels[j];
        if (groupOf[labels[j]] === g) continue;
        bgSum[g] += 0.212671 * SRGB_LIN[p & 255] + 0.715160 * SRGB_LIN[(p >> 8) & 255] + 0.072169 * SRGB_LIN[(p >> 16) & 255];
        bgN[g]++;
      }
    }
    const neutral = camConditions([95.047, 100, 108.883], 40, 20);
    for (const g of shown) {
      if (medoids[g] < 0) continue;
      const Yb = bgN[g] ? Math.min(100, Math.max(1, (bgSum[g] / bgN[g]) * scale)) : 20;
      display[g] = perceive(pixels[medoids[g]], camConditions(white.map((v) => v * scale), 40, Yb), neutral, scale);
    }
  }

  // 5. paint every pixel its medoid's color, and a debug version with one random color per cluster
  status("Rendering…", 0.96);
  const paletteBuf = buffer(K * 4, S.STORAGE | S.COPY_DST);
  const outBuf = buffer(n * 4, S.STORAGE | S.COPY_SRC);
  const renderGroup = bind("render", [labelBuf, paletteBuf, outBuf, paramBuf]);
  const paint = async (palette) => {
    device.queue.writeBuffer(paletteBuf, 0, palette);
    run("render", renderGroup, gx, gy);
    return new ImageData(await read(outBuf, n * 4, Uint8ClampedArray), w, h);
  };
  const debugRandom = rng(1);
  views = {
    original: new ImageData(new Uint8ClampedArray(pixels.buffer), w, h),
    medoid: await paint(Uint32Array.from(groupOf, (g) => (medoids[g] < 0 ? 0xff000000 : display[g] | 0xff000000))),
    debug: await paint((() => {
      const random = Array.from({ length: K }, () => (Math.floor(debugRandom() * 0xffffff) | 0xff000000) >>> 0);
      return Uint32Array.from(groupOf, (g) => random[g]);
    })()),
  };
  showView();
  $("debug").disabled = $("posterize").disabled = false;

  for (const b of [paramBuf, pxBuf, sharpBuf, featBuf, labelBuf, distBuf, centerBuf, sumBuf, medBuf, histBuf, workBuf, workLabBuf, filteredBuf, fBuf, paletteBuf, outBuf]) b.destroy();
  showChips(shown.filter((g) => medoids[g] >= 0).map((g) => ({ x: medoids[g] % w, y: Math.floor(medoids[g] / w), rgba: display[g] })), w, h);
  status(`${w}×${h} · ${seg.regions} regions · ${shown.length} colors ·${((performance.now() - t0) / 1000).toFixed(2)}s`);
}

// one slider per clustering weight; the image updates as the slider moves
for (const [name, value] of WEIGHTS) {
  const id = `weight-${name}`;
  const label = Object.assign(document.createElement("label"), { htmlFor: id, textContent: name });
  const input = Object.assign(document.createElement("input"), { id, type: "range", min: 0, max: 4, step: 0.1, value });
  const out = Object.assign(document.createElement("output"), { htmlFor: id, textContent: Number(value).toFixed(1) });
  input.addEventListener("input", () => { out.textContent = Number(input.value).toFixed(1); rerun(); });
  $("tuning").append(label, input, out);
}
// reset every setting to its default and rerun
const reset = Object.assign(document.createElement("button"), { type: "button", textContent: "Reset settings" });
reset.style.gridColumn = "1 / -1";
reset.addEventListener("click", () => {
  $("clusters").value = 5;
  $("group-count").value = 2;
  $("clusters-auto").checked = $("groups-auto").checked = true;
  $("perceived").checked = false;
  $("min-diff").value = $("min-diff-out").textContent = 20;
  $("spatial-radius").value = $("spatial-radius-out").textContent = 7;
  $("color-radius").value = $("color-radius-out").textContent = 8;
  $("group-spread").value = $("group-spread-out").textContent = 40;
  for (const [name, value] of WEIGHTS) {
    const input = $(`weight-${name}`);
    input.value = value;
    input.nextElementSibling.textContent = Number(value).toFixed(1);
  }
  rerun();
});
$("tuning").append(reset);
const weightValues = () => [...WEIGHTS.map(([name]) => Number($(`weight-${name}`).value)), 0, 0, 0];  // padded to 16

let views = null, debugMode = false, posterMode = false;
function showView() {
  if (!views) return;
  // original image by default; Posterized shows the medoid colors, Debug colors the random ones
  $("result").getContext("2d").putImageData(debugMode ? views.debug : posterMode ? views.medoid : views.original, 0, 0);
  $("debug").setAttribute("aria-pressed", String(debugMode));
  $("posterize").setAttribute("aria-pressed", String(posterMode));
}
// the two views are exclusive: turning one on turns the other off; both off shows the original
$("debug").addEventListener("click", () => { debugMode = !debugMode; if (debugMode) posterMode = false; showView(); });
$("posterize").addEventListener("click", () => { posterMode = !posterMode; if (posterMode) debugMode = false; showView(); });
$("perceived").addEventListener("change", rerun);
$("group-colors").addEventListener("click", (e) => {
  e.currentTarget.setAttribute("aria-pressed", String(e.currentTarget.getAttribute("aria-pressed") !== "true"));
  renderGroups();
});
$("settings").addEventListener("click", () => {
  const open = $("tuning").hidden;
  $("tuning").hidden = !open;
  $("settings").setAttribute("aria-pressed", String(open));
});


let current = null, hideTimer = 0;
function showChips(points, w, h) {
  points.sort((a, b) => a.y - b.y || a.x - b.x);  // top to bottom, so lines cross less
  $("chips").replaceChildren(...points.map((p) => {
    const el = document.createElement("div");
    const color = hex(p.rgba);
    const [r, g, b] = [p.rgba & 255, (p.rgba >> 8) & 255, (p.rgba >> 16) & 255];
    el.className = "chip";
    el.textContent = color;
    el.style.setProperty("--chip", color);
    el.style.setProperty("--chip-text", 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#000" : "#fff");  // fallback without contrast-color()
    return el;
  }));
  clearTimeout(hideTimer);
  current = { points, w, h };
  $("chips").style.visibility = $("lines").style.visibility = "";
  drawLines();
  renderGroups(points);
}


function swatch(rgba) {
  const el = document.createElement("div");
  const color = hex(rgba);
  const [r, g, b] = [rgba & 255, (rgba >> 8) & 255, (rgba >> 16) & 255];
  el.className = "chip";
  el.textContent = color;
  el.style.setProperty("--chip", color);
  el.style.setProperty("--chip-text", 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#000" : "#fff");
  return el;
}

// cluster clustering: group the medoid colors by Lab a, b only, then lay each group out as a gradient from dark to light
let lastColors = [];
function renderGroups(colors = lastColors) {
  lastColors = colors;
  if (!colors.length) return;
  // grouping uses HSV hue only: each color is a point on the hue circle (radius 180/pi, so nearby hues are ~1 apart per degree)
  const hsvHue = (rgba) => {
    const r = rgba & 255, g = (rgba >> 8) & 255, b = (rgba >> 16) & 255;
    const max = Math.max(r, g, b), d = max - Math.min(r, g, b);
    if (!d) return 0;
    const h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return (h * 60 + 360) % 360;
  };
  const labs = colors.map((p) => toLab(p.rgba));
  // plus LCh chroma at half weight as a third value
  const ab = colors.map((p, j) => {
    const t = (hsvHue(p.rgba) * Math.PI) / 180;
    return [(Math.cos(t) * 180) / Math.PI, (Math.sin(t) * 180) / Math.PI, 0.5 * Math.hypot(labs[j][1], labs[j][2])];
  });
  const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  const cluster = (k) => {
    const random = rng(2);
    const centers = [ab[Math.floor(random() * ab.length)]];
    while (centers.length < k) {  // k-means++ start
      const near = ab.map((p) => Math.min(...centers.map((c) => d2(p, c))));
      let r = random() * near.reduce((s, v) => s + v, 0), pick = 0;
      while (pick < ab.length - 1 && (r -= near[pick]) > 0) pick++;
      centers.push(ab[pick]);
    }
    let assign = [];
    for (let it = 0; it < 50; it++) {
      assign = ab.map((p) => centers.reduce((best, c, i) => (d2(p, c) < d2(p, centers[best]) ? i : best), 0));
      centers.forEach((_, i) => {
        const m = ab.filter((__, j) => assign[j] === i);
        if (m.length) centers[i] = [0, 1, 2].map((d) => m.reduce((s, p) => s + p[d], 0) / m.length);
      });
    }
    return { centers, assign };
  };
  // complete linkage: start with every color alone, keep merging the two groups whose farthest pair of colors is
  // closest, as long as that pair is within the limit, so no two colors in a group are farther apart than the limit
  const completeLinkage = (limit) => {
    const groups = ab.map((_, i) => [i]);
    const span = (g, h) => Math.max(...g.flatMap((i) => h.map((j) => Math.sqrt(d2(ab[i], ab[j])))));
    for (;;) {
      let best = Infinity, bi = -1, bj = -1;
      for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
        const s = span(groups[i], groups[j]);
        if (s < best) { best = s; bi = i; bj = j; }
      }
      if (bi < 0 || best > limit) break;
      groups[bi] = groups[bi].concat(groups[bj]);
      groups.splice(bj, 1);
    }
    const assign = new Array(ab.length);
    groups.forEach((g, gi) => g.forEach((i) => (assign[i] = gi)));
    const centers = groups.map((g) => [0, 1, 2].map((d) => g.reduce((s, i) => s + ab[i][d], 0) / g.length));
    return { centers, assign };
  };
  let k = Math.min(colors.length, Math.max(1, Math.round(Number($("group-count").value)) || 2));
  let result;
  if ($("groups-auto").checked) {  // auto: as many groups as complete linkage leaves at the group spread
    result = completeLinkage(Number($("group-spread").value));
    k = result.centers.length;
    $("group-count").value = k;
  } else {
    result = cluster(k);
  }
  const { centers, assign } = result;
  const groups = centers
    .map((c, i) => ({ hue: Math.atan2(c[1], c[0]), center: c, members: colors.map((p, j) => ({ p, ab: ab[j], lab: labs[j], L: labs[j][0], g: assign[j] })).filter((m) => m.g === i) }))
    .filter((g) => g.members.length)
    .sort((a, b) => a.hue - b.hue);  // rows ordered by hue
  // each group as a full-width gradient along its top principal component (PCA on the members' Lab colors),
  // with the hex codes of its two endpoints on the left and right
  const fromHex = (s) => parseInt(s.slice(1, 3), 16) | (parseInt(s.slice(3, 5), 16) << 8) | (parseInt(s.slice(5, 7), 16) << 16);
  const showMembers = $("group-colors").getAttribute("aria-pressed") === "true";
  $("groups-out").replaceChildren(...groups.map((g) => {
    // optional: the group's colors as a row of swatches (dark to light) above its gradient
    const row = document.createElement("div");
    row.className = "gradient";
    row.append(...[...g.members].sort((a, b) => a.L - b.L).map((m) => swatch(m.p.rgba)));
    const box = document.createElement("div");
    box.className = "group";
    const { css, lo, hi } = pcaGradient(g.members.map((m) => m.lab));
    const bar = document.createElement("div");
    bar.className = "pca";
    bar.style.background = css;
    box.append(swatch(fromHex(lo)), bar, swatch(fromHex(hi)));
    const wrap = document.createElement("div");  // one block per group, so the swatches sit flush on the gradient
    if (showMembers) wrap.append(row);
    wrap.append(box);
    return wrap;
  }));
  drawLines();
}
$("group-count").addEventListener("input", () => { $("groups-auto").checked = false; renderGroups(); });
$("groups-auto").addEventListener("change", () => renderGroups());
for (const id of ["min-diff", "spatial-radius", "color-radius", "group-spread"]) $(id).addEventListener("input", () => ($(`${id}-out`).textContent = $(id).value));
$("min-diff").addEventListener("input", () => { if ($("clusters-auto").checked) rerun(); });
for (const id of ["spatial-radius", "color-radius"]) $(id).addEventListener("input", rerun);
$("group-spread").addEventListener("input", () => { if ($("groups-auto").checked) renderGroups(); });

function drawLines() {
  const svg = $("lines");
  if (!current) return;
  const box = $("main").getBoundingClientRect(), img = $("result").getBoundingClientRect();
  const chips = [...$("chips").children];
  const NS = "http://www.w3.org/2000/svg";
  const shape = (tag, attrs) => { const el = document.createElementNS(NS, tag); for (const k in attrs) el.setAttribute(k, attrs[k]); return el; };
  // hex codes on the right connect from their left edge to their medoid pixel
  const pairs = current.points.map((p, i) => [chips[i], p, "left"]);
  svg.replaceChildren(...pairs.flatMap(([el, p, edge]) => {
    const chip = el.getBoundingClientRect();
    const x1 = chip[edge] - box.left, y1 = chip.top + chip.height / 2 - box.top;
    const x2 = img.left - box.left + (p.x + 0.5) * img.width / current.w, y2 = img.top - box.top + (p.y + 0.5) * img.height / current.h;
    const line = { x1, y1, x2, y2 };
    return [
      shape("line", { ...line, stroke: "rgba(0,0,0,0.55)", "stroke-width": 3 }),  // dark halo so the line reads on any color
      shape("line", { ...line, stroke: "#fff", "stroke-width": 1.25 }),
      shape("circle", { cx: x2, cy: y2, r: 4, fill: hex(p.rgba), stroke: "#fff", "stroke-width": 1.5 }),
    ];
  }));
}
new ResizeObserver(drawLines).observe(document.body);

// reruns are coalesced: while one is running, further changes just ask for one more run with the latest settings
let lastBitmap = null, running = false, pending = false;
async function rerun() {
  if (!lastBitmap) return;
  if (running) { pending = true; return; }
  running = true;
  try {
    do { pending = false; await recolor(lastBitmap); } while (pending);
  } catch (err) {
    console.error(err);
    status(`Error: ${err.message}`);
  } finally {
    running = false;
  }
}
$("clusters").addEventListener("input", () => { $("clusters-auto").checked = false; rerun(); });
$("clusters-auto").addEventListener("change", rerun);

async function handle(source) {
  try {
    if (!device) await initGPU();
    status("Loading image…", 0);
    lastBitmap = await createImageBitmap(source);
  } catch (err) {
    console.error(err);
    status(`Error: ${err.message}`);
    return;
  }
  rerun();
}

$("upload").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => { if (e.target.files[0]) { $("sample").hidden = true; handle(e.target.files[0]); } });
const main = $("main");
main.addEventListener("dragover", (e) => { e.preventDefault(); main.classList.add("dragging"); });
main.addEventListener("dragleave", () => main.classList.remove("dragging"));
main.addEventListener("drop", (e) => {
  e.preventDefault(); main.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) { $("sample").hidden = true; handle(file); }
});
// open on a built-in sample image (marked "Sample image" until the viewer uploads their own)
fetch("sample.jpg").then((r) => r.blob()).then(handle);
