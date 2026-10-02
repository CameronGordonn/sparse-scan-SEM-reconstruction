// TV-L2 inpainting in a Web Worker: a port of semrecon.baselines.tv.pdhg (data="l2", box=True),
// warm-started from a nearest-measured-pixel fill as in tv_inpaint.
//   min_x  lam * TV(x) + 1/2 sum_{i in M} (x_i - y_i)^2,   0 <= x <= 1
// Chambolle-Pock with forward differences and Neumann boundaries, tau = sigma = 0.99 / sqrt(8).

function nearestFill(y, mask, H, W) {
  // multi-source BFS from the measured pixels (4-neighbour; the Python version uses an exact EDT,
  // which only changes the warm start, not the optimum)
  const x = new Float32Array(H * W), seen = new Uint8Array(H * W), queue = new Int32Array(H * W);
  let head = 0, tail = 0;
  for (let i = 0; i < H * W; i++) if (mask[i]) { x[i] = y[i]; seen[i] = 1; queue[tail++] = i; }
  if (tail === 0) return x;
  while (head < tail) {
    const i = queue[head++], r = (i / W) | 0, c = i - r * W;
    const nb = [r > 0 ? i - W : -1, r < H - 1 ? i + W : -1, c > 0 ? i - 1 : -1, c < W - 1 ? i + 1 : -1];
    for (const j of nb) if (j >= 0 && !seen[j]) { seen[j] = 1; x[j] = x[i]; queue[tail++] = j; }
  }
  return x;
}

function pdhg(y, mask, H, W, lam, maxIter = 200, tol = 1e-4) {
  const N = H * W, tau = 0.99 / Math.sqrt(8), sigma = tau;
  let x = nearestFill(y, mask, H, W);
  const xbar = Float32Array.from(x), p0 = new Float32Array(N), p1 = new Float32Array(N);
  const xn = new Float32Array(N);
  for (let it = 0; it < maxIter; it++) {
    // p <- proj_{|p_i| <= lam}(p + sigma * grad xbar)
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        const g0 = r < H - 1 ? xbar[i + W] - xbar[i] : 0;
        const g1 = c < W - 1 ? xbar[i + 1] - xbar[i] : 0;
        const a = p0[i] + sigma * g0, b = p1[i] + sigma * g1;
        const s = Math.max(1, Math.sqrt(a * a + b * b) / lam);
        p0[i] = a / s; p1[i] = b / s;
      }
    }
    // x_new <- clip(prox_{tau G}(x + tau div p), 0, 1)
    let diff = 0, norm = 0;
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        let d = 0;
        if (r < H - 1) d += p0[i];
        if (r > 0) d -= p0[i - W];
        if (c < W - 1) d += p1[i];
        if (c > 0) d -= p1[i - 1];
        let v = x[i] + tau * d;
        if (mask[i]) v = (v + tau * y[i]) / (1 + tau);
        v = v < 0 ? 0 : v > 1 ? 1 : v;
        xn[i] = v;
        diff += (v - x[i]) ** 2; norm += x[i] ** 2;
      }
    }
    for (let i = 0; i < N; i++) { xbar[i] = 2 * xn[i] - x[i]; x[i] = xn[i]; }
    if (Math.sqrt(diff) / Math.max(Math.sqrt(norm), 1e-12) < tol) break;
  }
  return x;
}

if (typeof self !== "undefined") self.onmessage = (e) => {
  const { y, mask, H, W, lam, id } = e.data;
  const t = performance.now();
  const x = pdhg(y, mask, H, W, lam);
  self.postMessage({ x, id, ms: performance.now() - t }, [x.buffer]);
};

if (typeof module !== "undefined") module.exports = { pdhg, nearestFill };
