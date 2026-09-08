// Run with: node tests/gui_renderer_check.cjs (no browser or dependencies).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../src/mnt/nanoplacer/static/gui.js"), "utf8");
const renderer = source.slice(source.indexOf("  function cellCenter("), source.indexOf("  function zoom("));

for (const turning of [false, true]) {
  for (let rotation = 0; rotation < 4; rotation++) {
    const rotate = ([x, y, z]) => {
      for (let turn = 0; turn < rotation; turn++) [x, y] = [2 - y, x];
      return [x, y, z];
    };
    const cells = [0, 1].map(z => ({x: 1, y: 1, z, type: "buf", phase: 1}));
    const edges = [
      {source: [1, 0, 0], target: [1, 1, 1]},
      {source: [1, 1, 1], target: turning ? [2, 1, 0] : [1, 2, 0]},
      {source: [0, 1, 0], target: [1, 1, 0]},
      {source: [1, 1, 0], target: turning ? [1, 2, 0] : [2, 1, 0]},
    ].map(edge => ({source: rotate(edge.source), target: rotate(edge.target)}));
    const strokes = [];
    let points, dash;
    const context = {
      setTransform() {}, clearRect() {}, translate() {}, scale() {}, fillRect() {}, fillText() {},
      fill() {}, closePath() {}, strokeRect() {},
      setLineDash(value) { dash = value; },
      beginPath() { points = []; },
      moveTo(x, y) { points.push([x, y]); },
      lineTo(x, y) { points.push([x, y]); },
      arc(x, y, radius) { points.push([x, y, radius]); },
      stroke() { strokes.push({points, dash}); },
    };
    vm.runInNewContext(`${renderer}\ndraw();`, {
      preview: {width: 3, height: 3, cells, edges}, context, canvas: {},
      byId: () => ({getBoundingClientRect: () => ({width: 144, height: 144})}),
      window: {devicePixelRatio: 1}, view: {x: 0, y: 0, scale: 1}, selectedTile: null,
      tileSize: 48, phaseColors: ["#f0f4fb"], gateColors: {},
      cellsByTile: new Map([["1,1", cells]]),
    });
    // Four actual connections, all on-grid; ground drawn before upper. One shared dot.
    const center = ([x, y]) => [x * 48 + 24, y * 48 + 24];
    const ordered = [edges[2], edges[3], edges[0], edges[1]];
    assert.deepEqual(JSON.parse(JSON.stringify(strokes)), [
      ...ordered.map((edge, index) => ({
        points: [center(edge.source), center(edge.target)], dash: index < 2 ? [] : [4, 3],
      })),
      {points: [[72, 72, 5]], dash: []},
    ]);
  }
}
console.log("PASS centered single-dot crossings: solid/dashed layers, straight/turning wires in all four orientations");

const rewardRenderer = source.slice(source.indexOf("  function drawReward("), source.indexOf("  new ResizeObserver(drawReward)"));
for (const values of [[], [0], [-3, -1, 2], [5, 5, 5]]) {
  const strokes = [];
  const context = {
    setTransform() {}, beginPath() {}, closePath() {}, stroke() {}, fill() {}, fillText() {},
    moveTo(...point) { strokes.push(point); }, lineTo(...point) { strokes.push(point); },
    arc(...point) { strokes.push(point); },
    createLinearGradient: () => ({addColorStop() {}}),
  };
  const elements = new Map();
  const byId = id => {
    if (!elements.has(id)) elements.set(id, {
      getContext: () => context,
      getBoundingClientRect: () => ({width: 230, height: 190}),
    });
    return elements.get(id);
  };
  vm.runInNewContext(`${rewardRenderer}\ndrawReward();`, {
    byId, activeStates: new Set(["running"]), count: String, window: {devicePixelRatio: 2},
    run: {status: "running", timesteps: 100, mean_reward: values.at(-1) ?? null,
      reward_window: values.length, reward_history: values.map((value, index) => [index + 1, value])},
  });
  assert.equal(byId("reward-canvas").hidden, values.length === 0);
  assert.equal(byId("reward-empty").hidden, values.length > 0);
  assert.ok(strokes.every(point => point.every(Number.isFinite)));
  assert.equal(byId("reward-value").textContent, values.length ? String(values.at(-1)) : "—");
  if (!values.length) assert.match(byId("reward-empty").textContent, /first completed episode/);
}
console.log("PASS reward chart: waiting state, zero, negative, flat and single-point rewards at mobile width");

(async () => {
  const statusRenderer = source.slice(source.indexOf("  function renderStatus("), source.indexOf("  function drawReward("));
  const elements = new Map();
  const byId = id => {
    if (!elements.has(id)) elements.set(id, {textContent: "", dataset: {}});
    return elements.get(id);
  };
  const nextPreview = {width: 3, height: 4, cells: [{x: 0, y: 0, z: 0, type: "buf"}]};
  let resolvePreview, rejectPreview;
  const state = {
    byId, statusReady: true, run: null, seenRunId: "run-a", canResume: false,
    preview: {width: 3, height: 4, cells: [{x: 0, y: 0, z: 0, type: "pi", name: "a"}]},
    previewKey: "run-a:1", previewRequest: null, selectedTile: {x: 0, y: 0}, canvas: {}, cellsByTile: new Map(),
    activeStates: new Set(["running"]), count: String, elapsed: String,
    drawReward() {}, updateSettings() {}, renderFiles() {}, fit() {}, drawSoon() {},
    api: () => new Promise((resolve, reject) => { resolvePreview = resolve; rejectPreview = reject; }),
  };
  vm.createContext(state);
  vm.runInContext(statusRenderer, state);
  const data = {run: {
    id: "run-a", status: "running", preview_revision: 2, solution_found: true,
    config: {benchmark: "TOY", function: "and", clocking_scheme: "2DDWave", technology: "Gate-level"},
    timesteps: 20, total_timesteps: 100,
  }};
  state.renderStatus(data);
  assert.equal(byId("preview-kind").textContent, "Best partial placement");
  rejectPreview(new Error("Temporary preview failure"));
  await new Promise(setImmediate);
  assert.equal(state.previewKey, "run-a:1");
  assert.equal(byId("preview-kind").dataset.complete, "false");

  state.renderStatus(data);
  byId("canvas-detail").textContent = '(0, 0) · phase 1 · pi “a” · layer 0';
  resolvePreview(nextPreview);
  await new Promise(setImmediate);
  assert.equal(state.previewKey, "run-a:2");
  assert.equal(byId("preview-kind").textContent, "Best complete layout");
  assert.equal(state.selectedTile, null);
  assert.equal(byId("canvas-detail").textContent, "Gate-level preview · read-only");
  console.log("PASS preview state: partial badge during pending/failed updates; accepted revision clears stale inspection");
})().catch(error => { console.error(error); process.exitCode = 1; });
