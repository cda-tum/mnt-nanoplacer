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
    const state = {
      preview: {width: 3, height: 3, cells, edges}, context, canvas: {getContext: () => context},
      byId: () => ({getBoundingClientRect: () => ({width: 144, height: 144})}),
      window: {devicePixelRatio: 1}, view: {x: 0, y: 0, scale: 1}, selectedTile: null,
      tileSize: 48, phaseColors: ["#f0f4fb"], gateColors: {},
      cellsByTile: new Map([["1,1", cells]]),
      changedCells: new Set(), changedEdges: new Set(),
    };
    vm.runInNewContext(`${renderer}\ndraw();`, state);
    // Four actual connections, all on-grid; ground drawn before upper. One shared dot.
    const center = ([x, y]) => [x * 48 + 24, y * 48 + 24];
    const ordered = [edges[2], edges[3], edges[0], edges[1]];
    assert.deepEqual(JSON.parse(JSON.stringify(strokes)), [
      ...ordered.map((edge, index) => ({
        points: [center(edge.source), center(edge.target)], dash: index < 2 ? [] : [4, 3],
      })),
      {points: [[72, 72, 5]], dash: []},
    ]);
    if (!turning && rotation === 0) {
      // Export uses its own fitted viewport, even when the interactive canvas is far off-screen.
      state.view = {x: 10000, y: 10000, scale: 6};
      state.output = {getContext: () => context};
      strokes.length = 0;
      vm.runInNewContext(`${renderer}\ndraw(output, {x: 24, y: 24, scale: 1}, {width: 192, height: 192}, 4096 / 192, true);`, state);
      assert.equal(state.output.width, 4096);
      assert.equal(state.output.height, 4096);
      assert.equal(strokes.length, 5);
    }
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
  const statusRenderer = source.slice(source.indexOf("  function previewId("), source.indexOf("  function drawReward("));
  const elements = new Map();
  const byId = id => {
    if (!elements.has(id)) elements.set(id, {textContent: "", dataset: {}, setAttribute() {}});
    return elements.get(id);
  };
  const nextPreview = {width: 3, height: 4, cells: [{x: 0, y: 0, z: 0, type: "buf"}]};
  let resolvePreview, rejectPreview;
  const state = {
    byId, statusReady: true, run: null, seenRunId: "run-a", canResume: false,
    preview: {width: 3, height: 4, cells: [{x: 0, y: 0, z: 0, type: "pi", name: "a"}]},
    previewKey: "run-a:1", previewRequest: null, selectedTile: {x: 0, y: 0}, canvas: {}, cellsByTile: new Map(),
    activeStates: new Set(["running"]), count: String, elapsed: String,
    document: {fullscreenEnabled: true}, replayFrame: null, changedCells: new Set(), changedEdges: new Set(),
    drawReward() {}, updateSettings() {}, renderFiles() {}, fit() {}, drawSoon() {},
    api: () => new Promise((resolve, reject) => { resolvePreview = resolve; rejectPreview = reject; }),
  };
  vm.createContext(state);
  vm.runInContext(statusRenderer, state);
  state.run = {
    id: "run-a", status: "running", preview_revision: 2, solution_found: true,
    config: {benchmark: "TOY", function: "and", clocking_scheme: "2DDWave", technology: "Gate-level"},
    timesteps: 20, total_timesteps: 100,
  };
  state.loadPreview();
  assert.equal(byId("preview-kind").textContent, "Placement · not verified");
  assert.equal(byId("export-png").disabled, true);
  rejectPreview(new Error("Temporary preview failure"));
  await new Promise(setImmediate);
  assert.equal(state.previewKey, "run-a:1");
  assert.equal(byId("preview-kind").dataset.complete, "false");
  assert.equal(byId("export-png").disabled, true);

  state.loadPreview();
  byId("canvas-detail").textContent = '(0, 0) · phase 1 · pi “a” · layer 0';
  resolvePreview(nextPreview);
  await new Promise(setImmediate);
  assert.equal(state.previewKey, "run-a:2");
  assert.equal(byId("preview-kind").textContent, "Verified layout");
  assert.equal(state.selectedTile, null);
  assert.equal(byId("canvas-detail").textContent, "Gate-level preview · read-only");
  assert.equal(byId("export-png").disabled, false);
  // A requested replay frame must not export the previous frame under the new frame's name.
  state.previewKey = "run-a:frame-0";
  state.replayFrame = 1;
  state.api = path => path.endsWith("frame=1")
    ? new Promise(resolve => { resolvePreview = resolve; }) : Promise.resolve(nextPreview);
  state.loadPreview();
  assert.equal(byId("export-png").disabled, true);
  resolvePreview(nextPreview);
  await new Promise(setImmediate);
  assert.equal(state.previewKey, "run-a:frame-1");
  assert.equal(byId("export-png").disabled, false);
  // Switching runs while a request is pending must not display the old run's frame.
  state.replayFrame = null;
  state.api = () => new Promise(resolve => { resolvePreview = resolve; });
  state.run = {...state.run, id: "run-b", preview_revision: 3};
  state.loadPreview();
  state.run = {...state.run, id: "run-c"};
  resolvePreview({...nextPreview, width: 99});
  await new Promise(setImmediate);
  assert.equal(state.preview.width, 3);
  console.log("PASS preview state: partial badge during pending/failed updates; accepted revision clears stale inspection");
})().catch(error => { console.error(error); process.exitCode = 1; });

const comparisonLogic = source.slice(source.indexOf("  function settingsKey("), source.indexOf("  function renderComparison("));
const comparison = {activeStates: new Set(["queued", "running", "cancelling"])};
vm.createContext(comparison);
vm.runInContext(comparisonLogic, comparison);
const sampleRun = (id, seed, overrides = {}) => ({
  id, status: "completed", verified_solution: true,
  config: {benchmark: "trindade16", function: "mux21", clocking_scheme: "2DDWave", technology: "Gate-level",
    layout_width: 4, layout_height: 3, time_steps: 10000, optimize: true, stop_on_solution: false, resume: false, seed},
  manifest: {created_at: id, circuit_sha256: "circuit-sha", python: "3.11.0", packages: {"mnt.nanoplacer": "0.3.1", pyfiction: "0.8.0"}, source_checkpoint: null},
  ...overrides,
});
const first = sampleRun("a", 42, {verified_solution: false});
const repeated = sampleRun("b", 42);
const otherSeed = sampleRun("c", 43);
assert.equal(comparison.comparisonGroups([first, repeated, otherSeed]).length, 1);
assert.deepEqual(JSON.parse(JSON.stringify(comparison.seedSummary([repeated, first, otherSeed,
  sampleRun("d", 44, {status: "cancelled"}), sampleRun("e", 45, {status: "failed"}), sampleRun("f", 46, {status: "queued"})]))),
{seeds: 2, verified: 1, repeats: 1, resumed: 0, cancelled: 1, failed: 1, pending: 1});
for (const changed of [
  {...first, config: {...first.config, time_steps: 20000}},
  {...first, config: {...first.config, stop_on_solution: true}},
  {...first, manifest: {...first.manifest, circuit_sha256: "new-circuit"}},
  {...first, manifest: {...first.manifest, packages: {...first.manifest.packages, pyfiction: "0.9.0"}}},
  {...first, id: "old", manifest: {}},
]) assert.equal(comparison.comparisonGroups([first, changed]).length, 2);
assert.equal(comparison.comparisonGroups([sampleRun("old-a", 42, {manifest: {}}), sampleRun("old-b", 43, {manifest: {}})]).length, 2);
const resumed = {...first, config: {...first.config, resume: true}, manifest: {...first.manifest,
  source_checkpoint: {run_id: "source", filename: "recovery.zip", sha256: "checkpoint-sha"}}};
assert.equal(comparison.comparisonGroups([resumed, {...resumed, manifest: {...resumed.manifest,
  source_checkpoint: {...resumed.manifest.source_checkpoint, sha256: "later-checkpoint"}}}]).length, 2);
assert.equal(comparison.seedSummary([resumed]).seeds, 0);
assert.equal(comparison.seedSummary([resumed]).resumed, 1);
console.log("PASS fair comparison: budgets, stopping rules, software/circuit provenance and checkpoint identity stay separate; repeated seeds are not independent trials");

const learningCode = source.slice(source.indexOf("  function renderTarget("), source.indexOf("  async function refreshHistory("));
const learningElements = new Map();
const learning = {activeStates: new Set(["starting", "queued", "running"]), count: value => value == null ? "—" : String(value), byId: id => {
  if (!learningElements.has(id)) learningElements.set(id, {dataset: {}});
  return learningElements.get(id);
}, run: {ppo_epochs: 0, ppo_epochs_total: 0, first_solution_ppo_epochs: 0, first_solution_ppo_epochs_total: 0,
  best_placed: 9, total_nodes: 9, placement_history: [[0, 0, 9], [10, 7, 9], [20, 9, 9], [30, 9, 9]],
  reported_dimensions: [4, 3], target_reproduced: false, config: {layout_width: 5, layout_height: 5}, best_metrics: {width: 4, height: 3}}};
vm.createContext(learning);
vm.runInContext(learningCode, learning);
learning.renderLearning();
learning.renderTarget();
assert.match(learning.byId("learning-summary").textContent, /before any PPO training epoch/);
assert.equal(learning.byId("placement-progress").value, 100);
assert.match(learning.byId("placement-summary").textContent, /step 20/);
assert.match(learning.byId("target-result").textContent, /not reproduced/);
assert.equal(learning.byId("target-result").dataset.reproduced, "false");
learning.run.first_solution_ppo_epochs_total = 40;
learning.run.ppo_epochs_total = 40;
learning.run.target_reproduced = true;
learning.run.target_equivalent = "WEAK";
learning.renderLearning();
learning.renderTarget();
assert.match(learning.byId("learning-summary").textContent, /previously trained checkpoint \(40 epochs\)/);
assert.match(learning.byId("target-result").textContent, /weak equivalence \(timing differs\) before post-optimization/);
learning.run = {config: {}};
learning.renderLearning();
learning.renderTarget();
assert.equal(learning.byId("ppo-epochs").textContent, "—");
assert.equal(learning.byId("placement-progress").value, 0);
assert.match(learning.byId("target-result").textContent, /not recorded for this older run/);
learning.run.status = "starting";
learning.renderLearning();
learning.renderTarget();
assert.match(learning.byId("target-result").textContent, /Waiting for reported-target/);
assert.match(learning.byId("learning-summary").textContent, /Waiting for PPO/);
console.log("PASS learning and reported-target display: pre-training versus resumed success, bounded placement progress and no inference from optimized area");

const continueCode = source.slice(source.indexOf('  byId("continue-run").addEventListener'), source.indexOf('  byId("replay-frame").addEventListener'));
const continueHandlers = {};
let copiedSettings;
const continuation = {run: {...first, checkpoint_available: true}, activeRun: null, batch: null, pending: false,
  activeStates: new Set(["running"]), resumeRunId: null,
  byId: id => ({addEventListener(event, callback) { continueHandlers[id] = callback; }, scrollIntoView() {}}),
  restoreConfig(settings) { copiedSettings = settings; }, message() {}, updateSettings() {},
};
vm.runInNewContext(continueCode, continuation);
continueHandlers["continue-run"]();
assert.equal(copiedSettings.resume_run_id, first.id);
assert.equal(copiedSettings.resume, true);
assert.equal(copiedSettings.seed_count, 1);
assert.equal(copiedSettings.time_steps, first.config.time_steps);
copiedSettings = null;
continuation.activeRun = {status: "running"};
continueHandlers["continue-run"]();
assert.equal(copiedSettings, null);
console.log("PASS continue selected experiment: pins the exact source and copies an additional budget without starting training");

const disabledRenderer = source.slice(source.indexOf("  function updateDisabled("), source.indexOf("  function restoreConfig("));
const elements = new Map();
const byId = id => {
  if (!elements.has(id)) elements.set(id, {dataset: {}, removeAttribute(name) { delete this[name]; }});
  return elements.get(id);
};
const controls = {byId, activeStates: new Set(["running"]), activeRun: {id: "live", status: "running", config: {function: "mux21", seed: 42}},
  run: {id: "history", status: "completed"}, selectedRunId: "history", batch: {remaining: 2}, fields: {},
  catalog: {}, statusReady: true, pending: false, cancelling: false, resume: {checked: false}, resumeChecking: false,
  resumeRunId: null, automaticSteps: {checked: true}, automaticBudgetResolved: true, gridTooSmall: () => false};
vm.runInNewContext(`${disabledRenderer}\nupdateDisabled();`, controls);
assert.equal(controls.fields.disabled, true);
assert.equal(byId("start-button").disabled, true);
assert.equal(byId("cancel-button").hidden, false);
assert.equal(byId("cancel-button").textContent, "Stop seed series");
assert.equal(byId("repeat-settings").disabled, true);
assert.equal(byId("active-run-note").hidden, false);
console.log("PASS history browsing: active worker still locks configuration and can be stopped");
controls.activeRun.status = "completed";
vm.runInNewContext(`${disabledRenderer}\nupdateDisabled();`, controls);
assert.equal(byId("start-button").disabled, true);
assert.equal(byId("cancel-button").hidden, false);
console.log("PASS seed handoff: pending seeds retain the active experiment lock");

const runRenderer = source.slice(source.indexOf("  function renderRun("), source.indexOf("  async function refreshHistory("));
const qualityFields = ["quality-dimensions", "quality-area", "quality-wires", "quality-first"];
const exportLinks = ["export-experiment", "export-rewards", "export-metadata"];
for (const id of qualityFields) byId(id).textContent = "Previous result";
for (const id of exportLinks) byId(id).href = "/api/export/old-run/metadata.json";
byId("experiment-exports").hidden = false;
vm.runInNewContext(`${runRenderer}\nrenderRun(null);`, {
  byId, run: {}, seenRunId: "old-run", drawReward() {}, clearPreview() {}, renderFiles() {}, updateSettings() {}, count: String, activeStates: new Set(),
});
for (const id of qualityFields) assert.equal(byId(id).textContent, "—");
for (const id of exportLinks) assert.equal(byId(id).href, undefined);
assert.equal(byId("experiment-exports").hidden, true);
assert.equal(byId("episode-summary").textContent, "Complete placement and verified equivalence are reported separately.");
console.log("PASS empty current run: historical quality metrics and download links are cleared");

(async () => {
  const resumeCheck = source.slice(source.indexOf("  function checkResume("), source.indexOf("  function updateDisabled("));
  const jobs = [], requests = [];
  const state = {byId, resume: {checked: true}, resumeRunId: null, resumeEpoch: 0, resumeKey: null, resumeTimer: null,
    canResume: false, resumeChecking: false, config: () => ({seed_count: 1}), updateDisabled() {},
    api: () => new Promise(resolve => requests.push(resolve)),
    window: {setTimeout(callback) { jobs.push(callback); return jobs.length; }, clearTimeout() {}},
  };
  vm.createContext(state);
  vm.runInContext(resumeCheck, state);
  state.checkResume();
  const older = jobs.shift()();
  state.checkResume(true);
  const newer = jobs.shift()();
  requests[1]({can_resume: true, source_run_id: "completed-run", filename: "model.zip"});
  await newer;
  requests[0]({can_resume: false, reason: "No checkpoint yet."});
  await older;
  assert.equal(state.canResume, true);
  assert.equal(state.resume.disabled, false);
  assert.equal(state.resume.checked, true);
  assert.match(byId("resume-hint").textContent, /Compatible saved checkpoint/);
  state.resumeRunId = "exact-source";
  state.checkResume(true);
  const missing = jobs.shift()();
  requests[2]({can_resume: false, reason: "The selected checkpoint is unavailable."});
  await missing;
  assert.equal(state.resume.checked, true, "An unavailable explicit source must not silently become a fresh run.");
  assert.equal(state.resume.disabled, false, "The user can still intentionally deselect resume.");
  console.log("PASS compatible resume: stale same-settings checks cannot overwrite the completed run's checkpoint");
})().catch(error => { console.error(error); process.exitCode = 1; });

(async () => {
  const budgetCode = source.slice(source.indexOf("  function gridTooSmall("), source.indexOf("  function updateSettings("));
  const settingsCode = source.slice(source.indexOf("  function updateSettings("), source.indexOf("  function checkResume("));
  const restoreCode = source.slice(source.indexOf("  function restoreConfig("), source.indexOf("  const count ="));
  const toggleCode = source.slice(source.indexOf('  automaticSteps.addEventListener("change"'), source.indexOf("  for (const element of [circuit"));
  const requests = [], handlers = {};
  const state = {
    byId, count: String, benchmark: {value: "collection"}, circuit: {value: "small"},
    steps: {value: 10000}, automaticSteps: {checked: true, addEventListener(type, callback) { handlers[type] = callback; }},
    circuitKey: null, circuitInfo: null, circuitError: null, circuitEpoch: 0,
    savedBudgetKey: null, automaticBudgetResolved: false,
    technology: {}, clocking: {}, minimum: {}, width: {value: 20}, height: {value: 20, setCustomValidity() {}},
    seed: {value: 42, setCustomValidity() {}}, seedCount: {value: 1}, stopOnSolution: {}, optimize: {}, resume: {checked: false},
    activeStates: new Set(["running"]), activeRun: null, run: null, batch: null, selectedRunId: null,
    fields: {}, catalog: {limits: {max_tiles: 16384}}, statusReady: true, pending: false, cancelling: false, resumeChecking: false, resumeRunId: null,
    numbers: {format: String}, preset: () => null, checkResume() {}, renderComparison() {},
    api: path => new Promise((resolve, reject) => requests.push({path, resolve, reject})),
    fillCircuits(value) { state.circuit.value = value; },
  };
  vm.createContext(state);
  vm.runInContext(`${budgetCode}\n${settingsCode}\n${restoreCode}\n${disabledRenderer}\n${toggleCode}`, state);
  state.updateSettings();
  state.updateSettings();
  assert.equal(requests.length, 1, "Status polling must reuse the circuit-size lookup.");
  assert.equal(byId("start-button").disabled, true);
  assert.equal(state.steps.disabled, true);
  requests[0].resolve({placement_nodes: 9, recommended_timesteps: 10000});
  await new Promise(setImmediate);
  assert.equal(state.steps.value, 10000);
  assert.equal(byId("start-button").disabled, false);
  assert.match(byId("network-hint").textContent, /9 placement nodes/);

  state.circuit.value = "larger";
  state.updateSettings();
  state.circuit.value = "largest";
  state.updateSettings();
  requests[2].resolve({placement_nodes: 200, recommended_timesteps: 200000});
  await new Promise(setImmediate);
  requests[1].resolve({placement_nodes: 50, recommended_timesteps: 50000});
  await new Promise(setImmediate);
  assert.equal(state.steps.value, 200000, "A stale circuit response must not replace the current recommendation.");
  assert.match(byId("network-hint").textContent, /200 placement nodes/);
  state.width.value = 10;
  state.height.value = 10;
  state.updateSettings();
  assert.equal(byId("start-button").disabled, true, "Automatic budgets cannot start with fewer tiles than placement nodes.");
  assert.equal(byId("dimension-warning").hidden, false);
  assert.match(byId("dimension-warning").textContent, /200 placement nodes.*grid has 100/);

  state.automaticSteps.checked = false;
  handlers.change();
  assert.equal(byId("start-button").disabled, true, "Manual budgets also reject grids that cannot contain the nodes.");
  assert.equal(state.steps.value, 200000);
  assert.equal(state.steps.disabled, false);
  state.height.value = 20;
  state.updateSettings();
  assert.equal(byId("start-button").disabled, false, "Exactly one tile per placement node passes the necessary capacity check.");
  assert.equal(byId("dimension-warning").hidden, true);
  state.steps.value = 34567;
  state.circuit.value = "manual";
  state.updateSettings();
  requests[3].resolve({placement_nodes: 40, recommended_timesteps: 40000});
  await new Promise(setImmediate);
  assert.equal(state.steps.value, 34567, "A manual override survives circuit changes.");
  state.automaticSteps.checked = true;
  handlers.change();
  assert.equal(state.steps.value, 40000);
  assert.equal(requests.length, 4, "Toggling automatic mode reuses the loaded recommendation.");

  state.restoreConfig({benchmark: "collection", function: "saved", time_steps: 12345, automatic_time_steps: true});
  assert.equal(byId("start-button").disabled, false, "A saved resolved budget remains usable during the lookup.");
  requests[4].resolve({placement_nodes: 50, recommended_timesteps: 50000});
  await new Promise(setImmediate);
  assert.equal(state.steps.value, 12345, "Restoring an automatic run preserves its actual saved budget.");
  assert.equal(state.config().automatic_time_steps, true);
  assert.match(byId("network-hint").textContent, /Saved budget: 12345/);
  state.restoreConfig({benchmark: "collection", function: "saved", time_steps: 6789});
  assert.equal(state.automaticSteps.checked, false, "Legacy runs restore as manual budgets.");
  assert.equal(state.steps.value, 6789);
  assert.equal(state.steps.disabled, false);

  state.circuit.value = "unavailable";
  state.automaticSteps.checked = true;
  state.updateSettings();
  requests[5].reject(new Error("Could not read this circuit."));
  await new Promise(setImmediate);
  assert.equal(byId("start-button").disabled, true);
  assert.match(byId("network-hint").textContent, /Turn off automatic budget/);
  state.automaticSteps.checked = false;
  handlers.change();
  assert.equal(byId("start-button").disabled, false, "A failed recommendation must not block manual budgets.");
  assert.equal(state.steps.value, 6789);
  console.log("PASS automatic budgets: network size, manual override, stale responses, saved budgets and failure fallback");
})().catch(error => { console.error(error); process.exitCode = 1; });
