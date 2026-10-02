// In-browser demo: simulate a sparse SEM scan of an image and reconstruct it with TV-L2 (Web Worker)
// and the trained U-Net (ONNX Runtime Web). Ports semrecon.patterns, semrecon.forward and the
// preprocessing of scripts/reconstruct.py. Nothing leaves the visitor's device.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";

const CROP = 512;          // largest area reconstructed (pixels per side)
const TRAIN_WIDTH = 1024;  // wider images are shrunk to the pixel scale the models were trained on
const MODEL_URL = "models/unet.onnx";
// validation-tuned TV-L2 strength, fixed-dwell regime (results/tv_lambdas.json)
const TV_LAMBDA = {
  uniform: { 0.05: 0.025, 0.1: 0.05, 0.15: 0.05, 0.2: 0.1, 0.3: 0.1 },
  partial_raster: { 0.05: 0.025, 0.1: 0.05, 0.15: 0.1, 0.2: 0.1, 0.3: 0.1 },
  line_hop: { 0.05: 0.025, 0.1: 0.05, 0.15: 0.1, 0.2: 0.1, 0.3: 0.1 },
};
// scan-time model of semrecon.patterns.ScanTiming, in microseconds
const DWELL = 1, SETTLE = 5, FLYBACK = 50;

const $ = (id) => document.getElementById(id);
const state = { full: null, crop: { x: 0, y: 0, w: 0, h: 0 }, running: false, pending: false };

// ------------------------------------------------------------------ random numbers

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
}

function poisson(lam, rand) {
  if (lam <= 0) return 0;
  if (lam > 30) return Math.max(0, Math.round(lam + Math.sqrt(lam) * gaussian(rand)));
  const L = Math.exp(-lam);
  let k = 0, p = 1;
  do { k++; p *= rand(); } while (p > L);
  return k - 1;
}

// sample `count` distinct integers from [0, n), sorted
function sampleSorted(n, count, rand) {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(rand() * (n - i));
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return Array.from(a.subarray(0, count)).sort((u, v) => u - v);
}

// ------------------------------------------------------------------ scan patterns (semrecon.patterns)

function composition(total, parts, rand, minPart) {
  if (parts === 1) return [total];
  const free = total - parts * minPart;
  const bars = sampleSorted(free + parts - 1, parts - 1, rand);
  const edges = [-1, ...bars, free + parts - 1];
  const out = [];
  for (let i = 1; i < edges.length; i++) out.push(edges[i] - edges[i - 1] - 1 + minPart);
  return out;
}

function makeMask(pattern, H, W, frac, rand) {
  const m = new Uint8Array(H * W);
  if (pattern === "uniform") {
    const n = Math.round(frac * H * W), idx = new Int32Array(H * W);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(rand() * (idx.length - i));
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      m[idx[i]] = 1;
    }
  } else if (pattern === "partial_raster") {  // one line at a random offset in each of n strata
    const n = Math.max(1, Math.round(frac * H));
    for (let k = 0; k < n; k++) {
      const lo = Math.floor((k * H) / n), hi = Math.max(Math.floor(((k + 1) * H) / n), lo + 1);
      const row = lo + Math.floor(rand() * (hi - lo));
      m.fill(1, row * W, row * W + W);
    }
  } else {  // line_hop: every line, random-length segments, forward hops over the gaps
    const k = Math.round(frac * W);
    const nseg = Math.min(Math.max(Math.round(k / 16), 1), W - k + 1);
    for (let r = 0; r < H; r++) {
      const seg = composition(k, nseg, rand, 1);
      const gaps = composition(W - k + 2, nseg + 1, rand, 1);
      gaps[0] -= 1; gaps[gaps.length - 1] -= 1;
      let pos = 0;
      for (let s = 0; s < nseg; s++) {
        pos += gaps[s];
        m.fill(1, r * W + pos, r * W + pos + seg[s]);
        pos += seg[s];
      }
    }
  }
  return m;
}

function scanTime(mask, H, W) {
  let pixels = 0, lines = 0, runs = 0;
  for (let r = 0; r < H; r++) {
    let rowRuns = 0, prev = 0;
    for (let c = 0; c < W; c++) {
      const v = mask[r * W + c];
      pixels += v;
      if (v && !prev) rowRuns++;
      prev = v;
    }
    if (rowRuns) { lines++; runs += rowRuns; }
  }
  const t = pixels * DWELL + (runs - lines) * SETTLE + lines * FLYBACK;
  return t / (H * W * DWELL + H * FLYBACK);
}

// ------------------------------------------------------------------ images

function psnr(pred, truth) {
  let s = 0;
  for (let i = 0; i < truth.length; i++) {
    const p = Math.min(1, Math.max(0, pred[i])), d = p - truth[i];
    s += d * d;
  }
  return 10 * Math.log10(1 / Math.max(s / truth.length, 1e-12));
}

function draw(canvas, data, H, W) {
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(W, H);
  for (let i = 0; i < H * W; i++) {
    const v = Math.round(Math.min(1, Math.max(0, data[i])) * 255);
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

let utifLoaded = null;
function loadUTIF() {
  utifLoaded ??= new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/utif@3.1.0/UTIF.js";
    s.onload = resolve; s.onerror = () => reject(new Error("could not load the TIFF decoder"));
    document.head.appendChild(s);
  });
  return utifLoaded;
}

// any image file -> canvas holding an 8-bit grayscale version at the training pixel scale
async function fileToCanvas(blob, name) {
  let source;
  if (/\.tiff?$/i.test(name) || blob.type === "image/tiff") {
    await loadUTIF();
    const buf = await blob.arrayBuffer();
    const ifd = UTIF.decode(buf)[0];
    UTIF.decodeImage(buf, ifd);
    const rgba = UTIF.toRGBA8(ifd);
    const c = new OffscreenCanvas(ifd.width, ifd.height);
    c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer), ifd.width, ifd.height), 0, 0);
    source = c;
  } else {
    source = await createImageBitmap(blob);
  }
  const scale = Math.min(1, TRAIN_WIDTH / source.width);
  const W = Math.round(source.width * scale), H = Math.round(source.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, W, H);
  // grayscale as PIL "L" (ITU-R 601-2), then stretch to the full range like 16-bit TIFFs in the CLI
  const img = ctx.getImageData(0, 0, W, H), gray = new Float32Array(W * H);
  let lo = 1, hi = 0;
  for (let i = 0; i < W * H; i++) {
    const g = (img.data[4 * i] * 299 + img.data[4 * i + 1] * 587 + img.data[4 * i + 2] * 114) / 255000;
    gray[i] = g; lo = Math.min(lo, g); hi = Math.max(hi, g);
  }
  const isTiff = source instanceof OffscreenCanvas, span = hi - lo;
  if (isTiff && span > 1e-6 && span < 0.9) for (let i = 0; i < gray.length; i++) gray[i] = (gray[i] - lo) / span;
  draw(canvas, gray, H, W);
  return { canvas, gray, H, W };
}

// ------------------------------------------------------------------ crop picker

function setImage(full) {
  state.full = full;
  const w = Math.min(CROP, full.W - (full.W % 16)), h = Math.min(CROP, full.H - (full.H % 16));
  state.crop = { x: Math.floor((full.W - w) / 2), y: Math.floor((full.H - h) / 2), w, h };
  const picker = $("picker");
  const needsPick = full.W > w || full.H > h;
  picker.hidden = !needsPick;
  if (needsPick) {
    const view = $("picker-image");
    view.width = full.W; view.height = full.H;
    view.getContext("2d").drawImage(full.canvas, 0, 0);
    placeBox();
  }
}

function placeBox() {
  const { full, crop } = state, box = $("picker-box");
  box.style.left = `${(100 * crop.x) / full.W}%`;
  box.style.top = `${(100 * crop.y) / full.H}%`;
  box.style.width = `${(100 * crop.w) / full.W}%`;
  box.style.height = `${(100 * crop.h) / full.H}%`;
}

function initPicker() {
  const box = $("picker-box"), frame = $("picker-frame");
  let start = null;
  box.addEventListener("pointerdown", (e) => {
    box.setPointerCapture(e.pointerId);
    start = { px: e.clientX, py: e.clientY, x: state.crop.x, y: state.crop.y };
  });
  box.addEventListener("pointermove", (e) => {
    if (!start) return;
    const k = state.full.W / frame.getBoundingClientRect().width;
    const { full, crop } = state;
    crop.x = Math.round(Math.min(full.W - crop.w, Math.max(0, start.x + (e.clientX - start.px) * k)));
    crop.y = Math.round(Math.min(full.H - crop.h, Math.max(0, start.y + (e.clientY - start.py) * k)));
    placeBox();
  });
  const end = () => { if (start) { start = null; schedule(); } };
  box.addEventListener("pointerup", end);
  box.addEventListener("pointercancel", end);
}

function cropped() {
  const { full, crop } = state, out = new Float32Array(crop.w * crop.h);
  for (let r = 0; r < crop.h; r++) {
    out.set(full.gray.subarray((crop.y + r) * full.W + crop.x, (crop.y + r) * full.W + crop.x + crop.w), r * crop.w);
  }
  return { x: out, H: crop.h, W: crop.w };
}

// ------------------------------------------------------------------ reconstruction

let session = null;
async function getSession(status) {
  if (session) return session;
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`could not download the model (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 31e6;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    status(`Downloading the neural network (31 MB, once)… ${Math.min(99, Math.round((100 * got) / total))}%`);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  status("Loading the neural network…");
  try {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu", "wasm"] });
  } catch {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  }
  return session;
}

async function runUNet(y, mask, H, W, status) {
  const s = await getSession(status);
  status("Running the neural network…");
  await new Promise(requestAnimationFrame);
  const m = Float32Array.from(mask);
  const t = performance.now();
  const out = await s.run({
    y: new ort.Tensor("float32", y, [1, 1, H, W]),
    mask: new ort.Tensor("float32", m, [1, 1, H, W]),
  });
  return { x: out.x.data, ms: performance.now() - t };
}

const worker = new Worker("tv-worker.js");
function runTV(y, mask, H, W, lam) {
  return new Promise((resolve) => {
    worker.onmessage = (e) => resolve(e.data);
    worker.postMessage({ y: Float32Array.from(y), mask: Uint8Array.from(mask), H, W, lam });
  });
}

function settings() {
  const pattern = document.querySelector('input[name="pattern"]:checked').value;
  const frac = Number($("frac").value) / 100;
  const dose = Number(document.querySelector('input[name="noise"]:checked').value);
  return { pattern, frac, dose };
}

function nearestLambda(pattern, frac) {
  const fr = Object.keys(TV_LAMBDA[pattern]).map(Number);
  const near = fr.reduce((a, b) => (Math.abs(b - frac) < Math.abs(a - frac) ? b : a));
  return TV_LAMBDA[pattern][near];
}

function setResult(id, meta) { $(`${id}-meta`).textContent = meta; }

async function run() {
  if (!state.full) return;
  if (state.running) { state.pending = true; return; }
  state.running = true;
  const status = (t) => { $("status").textContent = t; };
  try {
    do {
      state.pending = false;
      const { x, H, W } = cropped();
      const { pattern, frac, dose } = settings();
      const rand = mulberry32(1);
      const mask = makeMask(pattern, H, W, frac, rand);
      const y = new Float32Array(H * W);
      for (let i = 0; i < H * W; i++) if (mask[i]) y[i] = poisson(Math.max(0, x[i]) * dose, rand) / dose;

      $("results").hidden = false;
      draw($("c-original"), x, H, W);
      draw($("c-measured"), y, H, W);
      setResult("original", `${W} × ${H} pixels`);
      setResult("measured", `${Math.round(frac * 100)}% of pixels · ${scanTime(mask, H, W).toFixed(2)}× the time of a full scan`);
      for (const id of ["tv", "unet"]) { $(`c-${id}`).classList.add("stale"); setResult(id, "working…"); }

      const tv = runTV(y, mask, H, W, nearestLambda(pattern, frac)).then(({ x: xt, ms }) => {
        draw($("c-tv"), xt, H, W); $("c-tv").classList.remove("stale");
        setResult("tv", `${psnr(xt, x).toFixed(1)} dB · ${(ms / 1000).toFixed(1)} s`);
      });
      const un = await runUNet(y, mask, H, W, status);
      draw($("c-unet"), un.x, H, W); $("c-unet").classList.remove("stale");
      setResult("unet", `${psnr(un.x, x).toFixed(1)} dB · ${(un.ms / 1000).toFixed(1)} s`);
      status("Finishing the traditional method…");
      await tv;
      status("");
    } while (state.pending);
  } catch (err) {
    status(`Something went wrong: ${err.message}`);
    console.error(err);
  } finally {
    state.running = false;
  }
}

let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(run, 250); }

// ------------------------------------------------------------------ wiring

async function useBlob(blob, name) {
  $("status").textContent = "Reading the image…";
  try {
    const full = await fileToCanvas(blob, name);
    if (full.W < 64 || full.H < 64) throw new Error("the image is too small (under 64 pixels)");
    setImage(full);
    run();
  } catch (err) {
    $("status").textContent = `Could not read that image: ${err.message}`;
  }
}

$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) useBlob(f, f.name); });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  const f = e.dataTransfer.files[0]; if (f) useBlob(f, f.name);
});
for (const b of document.querySelectorAll("[data-example]")) {
  b.addEventListener("click", async () => {
    for (const o of document.querySelectorAll("[data-example]")) o.setAttribute("aria-pressed", o === b);
    const res = await fetch(`examples/${b.dataset.example}.png`);
    useBlob(await res.blob(), `${b.dataset.example}.png`);
  });
}
const PATTERN_HELP = {
  partial_raster: "Scan a subset of complete lines. The fastest pattern, and the best use of scan time in our tests.",
  line_hop: "Visit every line, but measure only short random stretches and hop over the gaps.",
  uniform: "Measure scattered random pixels. Common in research, but the beam has to jump and settle before almost every pixel, so it's slow.",
};
const showHelp = () => { $("pattern-help").textContent = PATTERN_HELP[settings().pattern]; };
$("frac").addEventListener("input", () => { $("frac-value").textContent = `${$("frac").value}%`; schedule(); });
for (const r of document.querySelectorAll('input[name="pattern"], input[name="noise"]')) {
  r.addEventListener("change", () => { showHelp(); schedule(); });
}
showHelp();
initPicker();
