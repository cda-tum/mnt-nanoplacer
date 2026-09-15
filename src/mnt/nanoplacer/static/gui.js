/* Native, read-only experiment UI. Run state and all reported metrics come from the server. */
"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId("run-form");
  const fields = byId("configuration-fields");
  const benchmark = byId("benchmark");
  const circuit = byId("function");
  const technology = byId("technology");
  const clocking = byId("clocking-scheme");
  const minimum = byId("minimum-dimensions");
  const width = byId("layout-width");
  const height = byId("layout-height");
  const steps = byId("time-steps");
  const automaticSteps = byId("automatic-time-steps");
  const seed = byId("seed");
  const seedCount = byId("seed-count");
  const stopOnSolution = byId("stop-on-solution");
  const optimize = byId("optimize");
  const resume = byId("resume");
  const canvas = byId("layout-canvas");
  const token = document.querySelector('meta[name="csrf-token"]').content;
  const activeStates = new Set(["queued", "starting", "running", "cancelling"]);
  const numbers = new Intl.NumberFormat();
  let catalog = null;
  let run = null;
  let activeRun = null;
  let selectedRunId = null;
  let recentRuns = [];
  let historyKey = null;
  let comparisonKey = null;
  let historyUpdated = 0;
  let selectionEpoch = 0;
  let batch = null;
  let seenRunId = null;
  let canResume = false;
  let resumeRunId = null;
  let resumeKey = null;
  let resumeEpoch = 0;
  let resumeTimer = null;
  let resumeChecking = false;
  let pending = false;
  let cancelling = false;
  let preview = null;
  let previewKey = null;
  let previewRequest = null;
  let fileKey = null;
  let connectionError = false;
  let statusEpoch = 0;
  let statusReady = false;
  let view = { scale: 1, x: 0, y: 0 };
  let drawFrame = null;
  let pointer = null;
  let selectedTile = null;
  let cellsByTile = new Map();
  let replayFrame = null;
  let replayTimer = null;
  let changedCells = new Set();
  let changedEdges = new Set();
  let pollTimer = null;
  let polling = false;
  let circuitKey = null;
  let circuitInfo = null;
  let circuitError = null;
  let circuitEpoch = 0;
  let savedBudgetKey = null;
  let automaticBudgetResolved = false;

  function message(text, kind = "error") {
    const element = byId("message");
    if (element.textContent !== text) element.textContent = text;
    element.dataset.kind = kind;
    element.hidden = !text;
  }

  async function api(path, body) {
    const options = { cache: "no-store", signal: AbortSignal.timeout(15000) };
    if (body !== undefined) {
      options.method = "POST";
      options.headers = { "Content-Type": "application/json", "X-Nanoplacer-Token": token };
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
    return data;
  }

  function fillSelect(select, values, preferred) {
    select.replaceChildren(...values.map((value) => new Option(value, value)));
    if (values.includes(preferred)) select.value = preferred;
  }

  function fillCircuits(preferred) {
    const values = catalog.benchmarks[benchmark.value] || [];
    fillSelect(circuit, values, preferred || (values.includes("mux21") ? "mux21" : values[0]));
  }

  function preset() {
    return catalog?.dimensions[clocking.value]?.[benchmark.value]?.[circuit.value];
  }

  function gridTooSmall() {
    return Boolean(circuitInfo && circuitInfo.placement_nodes > Number(width.value) * Number(height.value));
  }

  function updateCircuitBudget() {
    const key = JSON.stringify([benchmark.value, circuit.value]);
    steps.disabled = automaticSteps.checked;
    if (key !== circuitKey) {
      circuitKey = key;
      circuitInfo = null;
      circuitError = null;
      if (savedBudgetKey !== key) savedBudgetKey = null;
      automaticBudgetResolved = savedBudgetKey === key;
      const epoch = ++circuitEpoch;
      api(`/api/circuit?benchmark=${encodeURIComponent(benchmark.value)}&function=${encodeURIComponent(circuit.value)}`)
        .then(data => {
          if (epoch !== circuitEpoch) return;
          circuitInfo = data;
          if (automaticSteps.checked && savedBudgetKey !== key) steps.value = data.recommended_timesteps;
          automaticBudgetResolved = true;
        })
        .catch(error => {
          if (epoch === circuitEpoch) circuitError = error.message;
        })
        .finally(() => { if (epoch === circuitEpoch) updateSettings(); });
    }
    const saved = savedBudgetKey === key && automaticSteps.checked
      ? `Saved budget: ${count(Number(steps.value))} timesteps per seed. ` : "";
    byId("network-hint").textContent = circuitInfo
      ? `${saved}${count(circuitInfo.placement_nodes)} placement nodes, including inputs, outputs and fanout nodes. Suggested budget: ${count(circuitInfo.recommended_timesteps)} timesteps per seed. A starting heuristic, not a guarantee of success.`
      : circuitError
        ? `${saved}Network size unavailable: ${circuitError} ${automaticSteps.checked && !automaticBudgetResolved ? "Turn off automatic budget to enter timesteps manually." : "Your entered budget is unchanged."}`
        : `${saved}Reading the input network to suggest a timestep budget…`;
  }

  function updateSettings() {
    if (!catalog) return;
    updateCircuitBudget();
    const sidb = technology.value === "SiDB";
    if (sidb) clocking.value = "2DDWave";
    clocking.disabled = sidb;
    byId("technology-hint").hidden = !sidb;
    optimize.disabled = clocking.value !== "2DDWave";
    if (optimize.disabled) optimize.checked = false;

    const dimensions = preset();
    minimum.disabled = !dimensions;
    if (!dimensions) minimum.checked = false;
    if (minimum.checked && dimensions) [width.value, height.value] = dimensions;
    width.disabled = minimum.checked;
    height.disabled = minimum.checked;
    byId("dimension-hint").textContent = dimensions
      ? `Reported minimum: ${dimensions[0]} × ${dimensions[1]} tiles. A starting point, not a guaranteed result.`
      : "No reported minimum for these settings. Choose your own width and height.";
    const area = Number(width.value) * Number(height.value);
    byId("tile-count").textContent = `${numbers.format(area || 0)} tiles`;
    const capacityWarning = gridTooSmall()
      ? `Grid too small: ${count(circuitInfo.placement_nodes)} placement nodes need at least that many tiles, but this grid has ${count(area)}. Increase width or height; routing may need additional space.` : "";
    byId("dimension-warning").textContent = capacityWarning;
    byId("dimension-warning").hidden = !capacityWarning;
    height.setCustomValidity(area > catalog.limits.max_tiles ? `Use at most ${numbers.format(catalog.limits.max_tiles)} tiles.` : capacityWarning);
    const seeds = Number(seedCount.value);
    seed.setCustomValidity(Number(seed.value) + seeds - 1 > 4294967295 ? "The final seed must not exceed 4294967295." : "");
    byId("budget-hint").textContent = `${count(seeds)} ${seeds === 1 ? "run" : "sequential runs"} · ${count(Number(steps.value) * seeds)} ${resume.checked ? "additional" : "total"} timesteps budget · seeds ${count(Number(seed.value))}${seeds > 1 ? `–${count(Number(seed.value) + seeds - 1)}` : ""}. Each run may finish its PPO rollout beyond the budget.`;
    checkResume();
    updateDisabled();
    renderComparison();
  }

  function checkResume(force = false) {
    const settings = config();
    const key = JSON.stringify({...settings, resume: false, stop_on_solution: false});
    if (!force && key === resumeKey) return;
    const epoch = ++resumeEpoch;
    resumeKey = key;
    window.clearTimeout(resumeTimer);
    canResume = false;
    resume.disabled = true;
    resumeChecking = settings.seed_count === 1;
    if (!resumeChecking) {
      resume.disabled = !resumeRunId;
      if (!resumeRunId) resume.checked = false;
    }
    byId("resume-hint").textContent = resumeChecking ? "Checking for a compatible checkpoint…" : "Multiple seeds start independently; resume is available for one seed only.";
    if (!resumeChecking) return;
    resumeTimer = window.setTimeout(async () => {
      try {
        const data = await api("/api/resume-check", {...settings, resume: false});
        if (resumeEpoch !== epoch) return;
        canResume = Boolean(data.can_resume);
        byId("resume-hint").textContent = canResume
          ? `Compatible ${data.checkpoint_kind === "recovery" ? "recovery" : "saved"} checkpoint: ${data.source_run_id.slice(0, 8)} / ${data.filename}. ${(data.warnings || []).join(" ")}`
          : data.reason || "No saved checkpoint matches this circuit, technology, clocking and grid.";
      } catch (error) {
        if (resumeEpoch !== epoch) return;
        byId("resume-hint").textContent = `Checkpoint check unavailable: ${error.message}`;
      } finally {
        if (resumeEpoch === epoch) {
          resumeChecking = false;
          resume.disabled = !canResume && !resumeRunId;
          if (!canResume && !resumeRunId) resume.checked = false;
          updateDisabled();
        }
      }
    }, 300);
  }

  function updateDisabled() {
    const active = Boolean(activeStates.has(activeRun?.status) || batch?.remaining);
    fields.disabled = !catalog || !statusReady || pending || active;
    byId("start-button").disabled = !catalog || !statusReady || pending || active || gridTooSmall() || (automaticSteps.checked && !automaticBudgetResolved) || (resume.checked && (resumeChecking || !canResume));
    byId("start-button").textContent = pending ? "Starting…" : active ? "Experiment in progress" : "Start experiment";
    byId("cancel-button").hidden = !active;
    byId("cancel-button").disabled = cancelling || activeRun?.status === "cancelling";
    byId("cancel-button").textContent = cancelling || activeRun?.status === "cancelling" ? "Stopping…" : batch?.remaining ? "Stop seed series" : "Stop active run";
    byId("repeat-settings").disabled = !run || active || pending;
    byId("continue-run").disabled = !run?.checkpoint_available || active || pending || activeStates.has(run?.status);
    byId("continue-run").title = run?.checkpoint_available ? "Copy these settings and continue this exact run's checkpoint." : "No saved checkpoint is available for this run.";
    byId("clear-resume-source").hidden = !resumeRunId;
    byId("active-run-note").hidden = !active || !selectedRunId || selectedRunId === activeRun.id;
    byId("active-run-note").textContent = active ? `Viewing a saved run. ${activeRun.config.function} · seed ${activeRun.config.seed} is ${activeRun.status}${batch?.remaining ? "; the seed series is still in progress" : ""}. Stop controls the active experiment, not the saved run.` : "";
  }

  function restoreConfig(config) {
    if (!config) return;
    benchmark.value = config.benchmark;
    fillCircuits(config.function);
    technology.value = config.technology;
    clocking.value = config.clocking_scheme;
    minimum.checked = Boolean(config.minimal_layout_dimension);
    width.value = config.layout_width;
    height.value = config.layout_height;
    steps.value = config.time_steps;
    automaticSteps.checked = Boolean(config.automatic_time_steps);
    savedBudgetKey = automaticSteps.checked ? JSON.stringify([benchmark.value, circuit.value]) : null;
    automaticBudgetResolved = automaticSteps.checked;
    seed.value = config.seed ?? 42;
    seedCount.value = config.seed_count ?? 1;
    stopOnSolution.checked = Boolean(config.stop_on_solution);
    optimize.checked = Boolean(config.optimize);
    resume.checked = Boolean(config.resume);
    resumeRunId = config.resume_run_id || null;
    updateSettings();
  }

  function config() {
    return {
      benchmark: benchmark.value,
      function: circuit.value,
      clocking_scheme: clocking.value,
      technology: technology.value,
      layout_width: Number(width.value),
      layout_height: Number(height.value),
      time_steps: Number(steps.value),
      automatic_time_steps: automaticSteps.checked,
      seed: Number(seed.value),
      seed_count: Number(seedCount.value),
      stop_on_solution: stopOnSolution.checked,
      optimize: optimize.checked,
      minimal_layout_dimension: minimum.checked,
      resume: resume.checked,
      resume_run_id: resumeRunId,
    };
  }

  const count = (value) => value == null ? "—" : numbers.format(value);
  function elapsed(value) {
    if (value == null) return "—";
    if (value < 10) return `${Math.max(0, value).toFixed(1)}s`;
    const seconds = Math.max(0, Math.floor(value));
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  }

  function renderFiles(files) {
    const key = JSON.stringify(files);
    if (key === fileKey) return;
    fileKey = key;
    const list = byId("files-list");
    list.replaceChildren();
    for (const file of files) {
      const url = new URL(file.url, window.location.href);
      if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/files/")) continue;
      const link = document.createElement("a");
      link.href = url.href;
      link.download = file.name;
      const name = document.createElement("span");
      name.className = "file-name";
      name.textContent = file.name;
      const size = document.createElement("span");
      size.className = "file-size";
      size.textContent = file.size >= 1048576 ? `${(file.size / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
      link.append(name, size);
      const item = document.createElement("li");
      item.append(link);
      list.append(item);
    }
    byId("file-count").textContent = `${list.children.length} files`;
    byId("files-empty").hidden = list.children.length > 0;
  }

  function clearPreview() {
    stopReplay();
    replayFrame = null;
    changedCells = new Set();
    changedEdges = new Set();
    preview = null;
    previewKey = null;
    selectedTile = null;
    cellsByTile = new Map();
    canvas.hidden = true;
    byId("preview-empty").hidden = false;
    byId("preview-empty-description").textContent = "The best observed placement will appear here. This animation is an illustration, not a training result.";
    byId("preview-kind").hidden = true;
    byId("preview-title").textContent = "A place for every gate";
    byId("preview-dimensions").textContent = "—";
    byId("canvas-detail").textContent = "Gate-level preview · read-only";
    byId("replay-controls").hidden = true;
    for (const id of ["zoom-in", "zoom-out", "fit-view", "fullscreen-view", "export-png"]) byId(id).disabled = true;
  }

  function renderStatus(data) {
    statusReady = true;
    const wasActive = activeStates.has(activeRun?.status);
    activeRun = data.run;
    batch = data.batch;
    byId("batch-summary").hidden = !batch;
    if (batch) byId("batch-summary").textContent = `Seed series · ${count(batch.completed)} / ${count(batch.total)} finished · ${count(batch.verified)} verified${batch.cancelled ? ` · ${count(batch.cancelled)} cancelled` : ""} · ${count(batch.remaining)} remaining.`;
    if (!selectedRunId || selectedRunId === activeRun?.id) renderRun(activeRun);
    if (wasActive && !activeStates.has(activeRun?.status)) checkResume(true);
    updateDisabled();
  }

  function renderRun(next) {
    run = next;
    drawReward();
    renderLearning();
    renderTarget();
    if (!run) {
      if (seenRunId !== null) {
        seenRunId = null;
        clearPreview();
        byId("run-title").textContent = "Ready when you are";
        byId("run-status").textContent = "Ready";
        byId("run-status").dataset.state = "idle";
        byId("run-description").textContent = "Choose a circuit and start an experiment.";
        byId("training-progress").value = 0;
        byId("progress-value").textContent = "—";
        byId("progress-label").textContent = "Training progress";
        for (const id of ["steps-value", "episodes-value", "elapsed-value", "placed-value", "quality-dimensions", "quality-area", "quality-wires", "quality-first"]) byId(id).textContent = "—";
        byId("episode-summary").textContent = "Complete placement and verified equivalence are reported separately.";
        byId("experiment-exports").hidden = true;
        for (const id of ["export-experiment", "export-rewards", "export-metadata"]) byId(id).removeAttribute("href");
        byId("run-outcome").textContent = "Start an experiment or select a saved run above.";
        byId("run-log").textContent = "Logs will appear after a run starts.";
        renderFiles([]);
      }
      updateSettings();
      return;
    }
    if (run.id !== seenRunId) {
      seenRunId = run.id;
      clearPreview();
      if (!selectedRunId) restoreConfig({...run.config,
        seed: run.config.seed - ((run.manifest?.batch_index || 1) - 1),
        seed_count: run.manifest?.batch_count || 1,
      });
    }
    updateSettings();
    const active = activeStates.has(run.status);
    const status = byId("run-status");
    status.textContent = run.status;
    status.dataset.state = run.status;
    const titles = { starting: "Preparing the agent", queued: "Preparing the agent", running: "Exploring the possibilities", cancelling: "Finishing this run", cancelled: "Experiment stopped", completed: "Experiment complete", failed: "Experiment needs attention" };
    byId("run-title").textContent = titles[run.status] || "Your experiment";
    byId("run-description").textContent = `${run.config.benchmark} / ${run.config.function} · ${run.config.clocking_scheme} · ${run.config.technology} · seed ${count(run.config.seed)}${run.manifest?.batch_count > 1 ? ` · run ${run.manifest.batch_index} / ${run.manifest.batch_count}` : ""}`;
    const actual = Number(run.timesteps) || 0;
    const total = Number(run.total_timesteps ?? run.config.time_steps) || 1;
    const percent = Math.min(100, Math.max(0, 100 * actual / total));
    byId("training-progress").value = percent;
    byId("progress-value").textContent = `${Math.floor(percent)}%`;
    byId("steps-value").textContent = count(run.timesteps);
    byId("episodes-value").textContent = count(run.episodes);
    byId("elapsed-value").textContent = elapsed(run.elapsed);
    byId("placed-value").textContent = run.total_nodes == null ? count(run.best_placed) : `${count(run.best_placed)} / ${count(run.total_nodes)}`;
    byId("progress-label").textContent = run.stop_reason === "solution" ? "Stopped early · verified solution found" : active && actual > total ? "Budget reached · finishing rollout" : `Budget · ${count(total)} timesteps`;
    let outcome;
    if (run.error) outcome = run.error;
    else if (run.verified_solution || run.solution_found) {
      const equivalence = {
        STRONG: "Complete layout found · strong equivalence verified.",
        WEAK: "Complete layout found · weak equivalence (timing differs).",
      };
      outcome = equivalence[run.equivalent] || "Complete layout found · equivalence not verified.";
    }
    else if (run.complete_candidate) outcome = "All gates placed, but no equivalent layout verified yet. A complete placement alone is not a solution.";
    else if (run.status === "cancelled") outcome = "Stopped. Available partial results and saved files are preserved.";
    else if (run.status === "completed") outcome = "Budget finished without a complete layout. Try another seed, more timesteps, or a larger grid.";
    else outcome = "Searching for a complete placement. The preview shows the best result observed so far.";
    if (byId("run-outcome").textContent !== outcome) byId("run-outcome").textContent = outcome;
    byId("preview-empty-description").textContent = active
      ? "Waiting for the first observed placement. This animation is an illustration, not a training result."
      : "No placement preview is available for this run. Check the training log for details.";
    const log = run.log || "No log output yet.";
    if (byId("run-log").textContent !== log) byId("run-log").textContent = log;
    renderFiles(run.files || []);
    const metrics = run.best_metrics || {};
    byId("quality-dimensions").textContent = metrics.width == null ? "—" : `${metrics.initial_width ?? run.config.layout_width} × ${metrics.initial_height ?? run.config.layout_height} → ${metrics.width} × ${metrics.height}`;
    byId("quality-area").textContent = count(metrics.area);
    byId("quality-wires").textContent = `${count(metrics.wires)} / ${count(metrics.crossings)}`;
    byId("quality-first").textContent = elapsed(run.first_solution_time);
    byId("episode-summary").textContent = run.complete_episodes == null ? "Episode quality statistics are available for new runs."
      : `Complete placements: ${count(run.complete_episodes)} · verified episodes: ${count(run.successful_episodes)}${run.episodes ? ` (${(100 * run.successful_episodes / run.episodes).toFixed(1)}%)` : ""} · routing failures: ${count(run.routing_failures)}.`;
    byId("experiment-exports").hidden = active;
    for (const [id, name] of [["export-experiment", "experiment.zip"], ["export-rewards", "rewards.csv"], ["export-metadata", "metadata.json"]]) {
      byId(id).href = `/api/export/${encodeURIComponent(run.id)}/${name}`;
    }
    const frames = Number(run.replay_count) || 0;
    byId("replay-controls").hidden = frames < 1;
    byId("replay-frame").max = Math.max(0, frames - 1);
    if (replayFrame === null) byId("replay-frame").value = Math.max(0, frames - 1);
    byId("replay-play").disabled = frames < 2;
    byId("replay-live").textContent = active ? "Live" : "Latest";
    loadPreview();
  }

  function renderTarget() {
    const target = byId("target-result");
    target.hidden = !run;
    target.dataset.reproduced = String(run?.target_reproduced === true);
    if (!run) return;
    const dimensions = run.reported_dimensions;
    if (!Array.isArray(dimensions)) {
      target.textContent = run.target_reproduced == null
        ? activeStates.has(run.status) ? "Waiting for reported-target verification data…" : "Reported-target proof was not recorded for this older run."
        : "No reported target dimensions for this circuit and clocking.";
    } else if (run.target_reproduced === true) {
      target.textContent = `Reported target reproduced · ${dimensions[0]} × ${dimensions[1]} tiles · ${run.target_equivalent === "WEAK" ? "weak equivalence (timing differs)" : "strong equivalence"} before post-optimization.`;
    } else {
      target.textContent = `Reported target: ${dimensions[0]} × ${dimensions[1]} tiles · ${run.target_reproduced == null ? "proof unavailable" : "not reproduced in this run"}. A smaller optimized result from another grid does not establish reproduction of this exact target.`;
    }
  }

  function renderLearning() {
    byId("ppo-epochs").textContent = count(run?.ppo_epochs);
    byId("ppo-epochs-total").textContent = count(run?.ppo_epochs_total);
    let learning = "Epochs count PPO optimization passes, not environment steps. Training episodes are online exploration, not an independent policy evaluation.";
    if (run?.ppo_epochs == null) learning = `${activeStates.has(run?.status) ? "Waiting for PPO learning statistics…" : "PPO learning statistics are available for new runs."} ${learning}`;
    else if (run.first_solution_ppo_epochs_total === 0) learning = `First verified solution was found before any PPO training epoch: successful exploration, not evidence of a learned policy. ${learning}`;
    else if (run.first_solution_ppo_epochs === 0 && run.first_solution_ppo_epochs_total > 0) learning = `First verified solution preceded this run's first PPO training epoch, using a previously trained checkpoint (${count(run.first_solution_ppo_epochs_total)} epochs). ${learning}`;
    else if (run.first_solution_ppo_epochs != null) learning = `First verified solution at step ${count(run.first_solution_timestep)}, after ${count(run.first_solution_ppo_epochs)} PPO training epochs in this run. ${learning}`;
    else if (run.ppo_epochs_total === 0) learning = `Collecting experience before the first PPO training epoch. ${learning}`;
    byId("learning-summary").textContent = learning;
    const history = (run?.placement_history || []).filter(point => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite));
    const latest = history.at(-1);
    const placed = latest?.[1] ?? run?.best_placed;
    const total = latest?.[2] ?? run?.total_nodes;
    const percent = total > 0 ? Math.min(100, Math.max(0, 100 * placed / total)) : 0;
    byId("placement-progress").value = percent;
    byId("placement-progress-value").textContent = total > 0 ? `${count(placed)} / ${count(total)} · ${Math.floor(percent)}%` : "—";
    const improvement = history.findLast((point, index) => index > 0 && point[1] > history[index - 1][1]);
    byId("placement-summary").textContent = `Best observed placement, not the current episode.${improvement ? ` Latest sampled improvement at step ${count(improvement[0])}.` : ""} Complete placement still requires equivalence verification.`;
    const checkpoint = run?.checkpoint;
    byId("checkpoint-summary").hidden = !checkpoint && !run?.checkpoint_error;
    byId("checkpoint-summary").textContent = `${checkpoint ? `${checkpoint.kind === "recovery" ? "Recovery" : "Final"} checkpoint saved at step ${count(checkpoint.timestep)} in this run (${count(checkpoint.total_timesteps)} lifetime). ` : ""}${run?.checkpoint_error || ""}`;
  }

  async function refreshHistory() {
    const data = await api("/api/runs");
    recentRuns = data.runs || [];
    historyUpdated = Date.now();
    const key = JSON.stringify(recentRuns);
    if (historyKey === key) return;
    historyKey = key;
    const selected = recentRuns.find(item => item.id === selectedRunId);
    if (selected && selected.id === run?.id && selected.id !== activeRun?.id && (selected.status !== run.status || selected.preview_revision !== run.preview_revision)) {
      const epoch = selectionEpoch;
      const detail = await api(`/api/runs/${encodeURIComponent(selected.id)}`);
      if (epoch === selectionEpoch) renderRun(detail.run);
    }
    const select = byId("run-history");
    select.replaceChildren(new Option("Latest experiment / follow active run", ""), ...recentRuns.map(item =>
      new Option(`${item.config.function} · seed ${item.config.seed} · ${item.status} · ${item.id.slice(0, 8)}`, item.id)));
    select.value = selectedRunId || "";
    renderComparison();
  }

  function settingsKey(settings) {
    return JSON.stringify([settings.benchmark, settings.function, settings.clocking_scheme, settings.technology,
      settings.layout_width, settings.layout_height, settings.time_steps, Boolean(settings.optimize),
      Boolean(settings.stop_on_solution), Boolean(settings.resume)]);
  }

  function provenanceKey(item) {
    const manifest = item.manifest || {};
    const packages = Object.entries(manifest.packages || {}).sort(([a], [b]) => a.localeCompare(b));
    const source = manifest.source_checkpoint;
    if (!manifest.circuit_sha256 || !manifest.python || !packages.length || packages.some(([, value]) => !value)
      || (item.config.resume && !source?.sha256)) return `unknown:${item.id}`;
    return JSON.stringify([manifest.circuit_sha256, manifest.python, packages,
      item.config.resume ? [source.run_id, source.filename, source.sha256] : null]);
  }

  function comparisonGroups(items) {
    const groups = new Map();
    for (const item of items) {
      const key = `${settingsKey(item.config)}:${provenanceKey(item)}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    return [...groups.values()];
  }

  function seedSummary(items) {
    const completed = items.filter(item => item.status === "completed" && !item.config.resume);
    const firstPerSeed = new Map();
    for (const item of [...completed].sort((a, b) => String(a.manifest?.created_at || "").localeCompare(String(b.manifest?.created_at || "")) || a.id.localeCompare(b.id))) {
      if (!firstPerSeed.has(item.config.seed)) firstPerSeed.set(item.config.seed, item);
    }
    const verified = [...firstPerSeed.values()].filter(item => item.verified_solution).length;
    return {
      seeds: firstPerSeed.size, verified, repeats: completed.length - firstPerSeed.size,
      resumed: items.filter(item => item.status === "completed" && item.config.resume).length,
      cancelled: items.filter(item => item.status === "cancelled").length,
      failed: items.filter(item => item.status === "failed").length,
      pending: items.filter(item => activeStates.has(item.status)).length,
    };
  }

  function renderComparison() {
    if (!catalog) return;
    const selected = selectedRunId && run?.id === selectedRunId ? run : null;
    const settings = selected?.config || config();
    const key = JSON.stringify([historyKey, selected?.id, settingsKey(settings), settings.resume_run_id, byId("matching-runs").checked]);
    if (key === comparisonKey) return;
    comparisonKey = key;
    const filtered = byId("matching-runs").checked ? recentRuns.filter(item =>
      settingsKey(item.config) === settingsKey(settings) && (!selected || provenanceKey(item) === provenanceKey(selected))
      && (selected || !settings.resume || !settings.resume_run_id || item.manifest?.source_checkpoint?.run_id === settings.resume_run_id)) : recentRuns;
    byId("comparison-filter-hint").textContent = selected
      ? `Matching ${selected.id.slice(0, 8)}: same circuit, grid, budget, optimization, stopping rule, training source and recorded circuit/software provenance.`
      : "Matching the form's circuit, grid, budget, optimization, stopping rule and fresh/resumed mode. Different software or source checkpoints stay in separate groups.";
    const groups = comparisonGroups(filtered);
    const summaries = groups.map((items, index) => {
      const summary = seedSummary(items);
      const item = items[0];
      const text = document.createElement("p");
      const config = item.config;
      const provenance = provenanceKey(item).startsWith("unknown:") ? "unknown provenance; not pooled" : `${item.manifest.circuit_sha256.slice(0, 8)} · NanoPlaceR ${item.manifest.packages["mnt.nanoplacer"] || "unknown"}`;
      text.textContent = `${groups.length > 1 ? `Group ${index + 1} · ` : ""}${config.function} · ${config.layout_width} × ${config.layout_height} · ${count(config.time_steps)} steps · ${provenance}: ${count(summary.verified)} / ${count(summary.seeds)} completed fresh seeds verified${summary.repeats ? ` (${count(summary.repeats)} repeat runs excluded)` : ""}; ${count(summary.resumed)} completed resumed · ${count(summary.cancelled)} cancelled · ${count(summary.failed)} failed · ${count(summary.pending)} pending/active.`;
      return text;
    });
    byId("comparison-summary").replaceChildren(...summaries);
    byId("comparison-caption").textContent = `Latest ${Math.min(10, filtered.length)} of ${count(filtered.length)} ${byId("matching-runs").checked ? "matching" : "saved"} runs. Summaries cover all listed history, grouped by settings and provenance; only the first completed fresh run per seed counts. Training success is not an independent policy evaluation.`;
    const rows = filtered.slice(0, 10).map(item => {
      const row = document.createElement("tr");
      const metrics = item.best_metrics || {};
      const values = [item.config.function, `${count(item.config.seed)} · ${item.config.resume ? "resumed" : "fresh"}`, `${count(item.timesteps)} / ${count(item.config.time_steps)}`, `${item.config.layout_width} × ${item.config.layout_height}${metrics.width == null ? "" : ` → ${metrics.width} × ${metrics.height}`}`, count(metrics.area), `${count(metrics.wires)} / ${count(metrics.crossings)}`, item.verified_solution || item.solution_found ? item.equivalent || "Yes" : item.complete_candidate ? "No" : "—", elapsed(item.first_solution_time), elapsed(item.elapsed)];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        if (index === 0) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = value;
          button.title = `${item.config.benchmark} · ${item.config.clocking_scheme} · ${item.config.technology} · ${item.id}`;
          button.addEventListener("click", () => selectRun(item.id));
          cell.append(button);
          const detail = document.createElement("small");
          detail.textContent = `${item.id.slice(0, 8)} · ${item.status} · ${item.config.clocking_scheme} · ${item.config.technology}`;
          cell.append(detail);
        } else {
          cell.textContent = value;
          if (index === 1 && item.config.resume) {
            const source = document.createElement("small");
            source.textContent = `from ${item.manifest?.source_checkpoint?.run_id?.slice(0, 8) || "unknown checkpoint"}`;
            cell.append(source);
          }
        }
        row.append(cell);
      });
      return row;
    });
    byId("comparison-body").replaceChildren(...rows);
    if (!rows.length) {
      const row = document.createElement("tr"), cell = document.createElement("td");
      cell.colSpan = 9;
      cell.textContent = "No matching experiments yet. Change the settings, select a saved run, or turn off the matching filter.";
      row.append(cell);
      byId("comparison-body").append(row);
    }
  }

  async function selectRun(id) {
    const epoch = ++selectionEpoch;
    selectedRunId = id || null;
    byId("run-history").value = id || "";
    stopReplay();
    try {
      const data = selectedRunId ? await api(`/api/runs/${encodeURIComponent(id)}`) : {run: activeRun};
      if (epoch !== selectionEpoch) return;
      renderRun(data.run);
      updateDisabled();
    } catch (error) {
      if (epoch === selectionEpoch) message(`Could not open this run. ${error.message}`);
    }
  }

  function previewId() {
    return run ? `${run.id}:${replayFrame === null ? run.preview_revision : `frame-${replayFrame}`}` : null;
  }

  function loadPreview() {
    updatePreviewLabel();
    const key = previewId();
    if (run?.preview_revision > 0 && key !== previewKey && key !== previewRequest) {
      previewRequest = key;
      const base = `/api/preview?run_id=${encodeURIComponent(run.id)}`;
      const requestedFrame = replayFrame;
      Promise.all([
        api(`${base}${requestedFrame === null ? `&revision=${run.preview_revision}` : `&frame=${requestedFrame}`}`),
        requestedFrame > 0 ? api(`${base}&frame=${requestedFrame - 1}`) : Promise.resolve(null),
      ]).then(([data, previous]) => {
        if (previewId() !== key) return;
        if (!data || !data.width || !data.height || !Array.isArray(data.cells)) return;
        const refit = !preview || preview.width !== data.width || preview.height !== data.height;
        changedCells = new Set();
        changedEdges = new Set();
        if (previous) {
          const oldCells = new Set(previous.cells.map(cell => JSON.stringify(cell)));
          const oldEdges = new Set((previous.edges || []).map(edge => JSON.stringify(edge)));
          for (const cell of data.cells) if (!oldCells.has(JSON.stringify(cell))) changedCells.add(`${cell.x},${cell.y},${cell.z}`);
          for (const edge of data.edges || []) if (!oldEdges.has(JSON.stringify(edge))) changedEdges.add(JSON.stringify(edge));
        }
        preview = data;
        previewKey = key;
        selectedTile = null;
        byId("canvas-detail").textContent = "Gate-level preview · read-only";
        cellsByTile = new Map();
        for (const cell of preview.cells) {
          const coordinate = `${cell.x},${cell.y}`;
          if (!cellsByTile.has(coordinate)) cellsByTile.set(coordinate, []);
          cellsByTile.get(coordinate).push(cell);
        }
        canvas.hidden = false;
        byId("preview-empty").hidden = true;
        byId("preview-kind").hidden = false;
        byId("preview-title").textContent = requestedFrame === null ? "The best so far" : "Replay the improvements";
        byId("preview-dimensions").textContent = `${data.width} × ${data.height} · ${data.clocking_scheme}`;
        for (const id of ["zoom-in", "zoom-out", "fit-view", "fullscreen-view", "export-png"]) byId(id).disabled = false;
        byId("fullscreen-view").hidden = !document.fullscreenEnabled;
        updatePreviewLabel();
        if (refit) fit(); else drawSoon();
      }).catch((error) => {
        if (previewId() === key) byId("canvas-detail").textContent = `Preview temporarily unavailable: ${error.message}`;
      }).finally(() => { if (previewRequest === key) previewRequest = null; });
    }
  }

  function updatePreviewLabel() {
    const current = previewKey === previewId();
    byId("export-png").disabled = !preview || !current;
    const complete = Boolean(current && (preview?.metadata?.verified ?? (replayFrame === null && (run?.verified_solution || run?.solution_found))));
    byId("preview-kind").textContent = complete ? "Verified layout" : "Placement · not verified";
    byId("preview-kind").dataset.complete = String(complete);
    const metadata = current ? preview?.metadata : null;
    byId("replay-summary").textContent = replayFrame === null
      ? `Latest best placement · ${count(run?.replay_count || 0)} saved improvements${run?.replay_truncated ? " (recording limit reached; latest result still updates)" : ""}.`
      : `Improvement ${replayFrame + 1} / ${count(run?.replay_count)}${metadata ? ` · step ${count(metadata.timestep)} · ${elapsed(metadata.elapsed)} · ${count(metadata.placed)} gates` : " · loading…"}. Orange marks changes from the previous saved improvement.`;
  }

  function stopReplay() {
    window.clearTimeout(replayTimer);
    replayTimer = null;
    byId("replay-play").textContent = "Play";
    byId("replay-play").setAttribute("aria-label", "Play saved improvements");
  }

  function showFrame(index) {
    replayFrame = index;
    byId("replay-frame").value = index ?? Math.max(0, (run?.replay_count || 1) - 1);
    loadPreview();
  }

  function drawReward() {
    const chart = byId("reward-canvas");
    const empty = byId("reward-empty");
    const points = (run?.reward_history || []).filter(point =>
      Array.isArray(point) && point.length === 2 && point.every(Number.isFinite) && point[0] >= 0);
    const mean = run?.mean_reward;
    const format = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});
    byId("reward-value").textContent = Number.isFinite(mean) ? format.format(mean) : "—";
    chart.hidden = points.length === 0;
    empty.hidden = points.length > 0;
    if (!points.length) {
      empty.textContent = !run ? "Start an experiment to see how the agent learns."
        : !("reward_history" in run) && run.status !== "starting" ? "Reward tracking is available for new experiments."
        : activeStates.has(run.status) ? "Waiting for the first completed episode…"
        : "No completed-episode rewards were recorded for this run.";
      byId("reward-summary").textContent = "Rewards are measured after each completed episode. History is sampled for display.";
      return;
    }
    byId("reward-summary").textContent = `Latest mean: ${format.format(mean)} over ${count(run.reward_window)} completed episodes. Sampled history · timesteps relative to this run.`;
    const ctx = chart.getContext("2d");
    const bounds = byId("reward-chart-wrap").getBoundingClientRect();
    if (!ctx || !bounds.width || !bounds.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    chart.width = Math.round(bounds.width * ratio);
    chart.height = Math.round(bounds.height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const left = 52, top = 12, right = bounds.width - 12, bottom = bounds.height - 38;
    const values = points.map(point => point[1]);
    const low = Math.min(...values), high = Math.max(...values);
    const padding = Math.max((high - low) * .12, Math.abs(high) * .05, .5);
    const min = low - padding, max = high + padding;
    const maxStep = Math.max(run.timesteps || 0, points.at(-1)[0], 1);
    const x = step => left + step / maxStep * (right - left);
    const y = reward => bottom - (reward - min) / (max - min) * (bottom - top);
    const compact = new Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1});
    ctx.font = "9px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.lineWidth = 1;
    for (let tick = 0; tick <= 3; tick++) {
      const reward = min + (max - min) * tick / 3;
      ctx.strokeStyle = "#edf0f7";
      ctx.beginPath();
      ctx.moveTo(left, y(reward));
      ctx.lineTo(right, y(reward));
      ctx.stroke();
      ctx.fillStyle = "#8491a6";
      ctx.textAlign = "right";
      ctx.fillText(compact.format(reward), left - 8, y(reward) + 3, left - 12);
    }
    for (let tick = 0; tick <= 2; tick++) {
      const step = Math.round(maxStep * tick / 2);
      ctx.textAlign = tick === 0 ? "left" : tick === 2 ? "right" : "center";
      ctx.fillText(compact.format(step), x(step), bottom + 15);
    }
    ctx.textAlign = "center";
    ctx.fillText("Timesteps in this run", (left + right) / 2, bounds.height - 3);
    const gradient = ctx.createLinearGradient(0, top, 0, bottom);
    gradient.addColorStop(0, "#7e8bea30");
    gradient.addColorStop(1, "#7e8bea03");
    ctx.beginPath();
    ctx.moveTo(x(points[0][0]), bottom);
    for (const [step, reward] of points) ctx.lineTo(x(step), y(reward));
    ctx.lineTo(x(points.at(-1)[0]), bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.beginPath();
    for (const [index, [step, reward]] of points.entries()) {
      if (index === 0) ctx.moveTo(x(step), y(reward));
      else ctx.lineTo(x(step), y(reward));
    }
    ctx.strokeStyle = "#647bdd";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x(points.at(-1)[0]), y(points.at(-1)[1]), 3, 0, Math.PI * 2);
    ctx.fillStyle = "#647bdd";
    ctx.fill();
  }

  new ResizeObserver(drawReward).observe(byId("reward-chart-wrap"));

  async function poll() {
    if (polling) return;
    polling = true;
    window.clearTimeout(pollTimer);
    const epoch = statusEpoch;
    try {
      const data = await api("/api/status");
      // A status request sent before Start/Stop must not overwrite its response.
      if (epoch !== statusEpoch || pending || cancelling) return;
      renderStatus(data);
      if (Date.now() - historyUpdated > (activeStates.has(activeRun?.status) ? 5000 : 15000)) await refreshHistory();
      if (connectionError) {
        connectionError = false;
        message("");
      }
    } catch (error) {
      connectionError = true;
      message(`Connection interrupted. Retrying automatically. ${error.message}`);
    } finally {
      polling = false;
      pollTimer = window.setTimeout(poll, document.hidden ? 5000 : activeStates.has(activeRun?.status) ? 750 : 3000);
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopReplay();
    else poll();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!catalog || !statusReady || pending || activeStates.has(activeRun?.status) || batch?.remaining) return;
    updateSettings();
    if (gridTooSmall() || (automaticSteps.checked && !automaticBudgetResolved)) return;
    if (!form.reportValidity()) return;
    if (Number(width.value) * Number(height.value) > catalog.limits.max_tiles) {
      message(`Use at most ${count(catalog.limits.max_tiles)} tiles.`);
      return;
    }
    pending = true;
    statusEpoch += 1;
    updateDisabled();
    message("");
    try {
      if (resume.checked) {
        const compatible = await api("/api/resume-check", config());
        if (!compatible.can_resume) throw new Error(compatible.reason || "No compatible checkpoint is available.");
      }
      selectedRunId = null;
      selectionEpoch += 1;
      byId("run-history").value = "";
      renderStatus(await api("/api/start", config()));
      historyUpdated = 0;
    } catch (error) {
      message(`Could not start the experiment. ${error.message}`);
    } finally {
      pending = false;
      statusEpoch += 1;
      updateDisabled();
    }
  });

  byId("cancel-button").addEventListener("click", async () => {
    if (cancelling || (!activeStates.has(activeRun?.status) && !batch?.remaining)) return;
    cancelling = true;
    statusEpoch += 1;
    updateDisabled();
    try {
      renderStatus(await api("/api/cancel", {}));
      historyUpdated = 0;
    } catch (error) {
      message(`Could not stop the experiment. ${error.message}`);
    } finally {
      cancelling = false;
      statusEpoch += 1;
      updateDisabled();
    }
  });

  benchmark.addEventListener("change", () => { fillCircuits(); updateSettings(); });
  automaticSteps.addEventListener("change", () => {
    savedBudgetKey = null;
    automaticBudgetResolved = Boolean(circuitInfo);
    if (automaticSteps.checked && circuitInfo) steps.value = circuitInfo.recommended_timesteps;
    updateSettings();
  });
  for (const element of [circuit, clocking, technology, minimum, optimize, stopOnSolution, resume]) element.addEventListener("change", updateSettings);
  for (const element of [width, height, steps, seed, seedCount]) element.addEventListener("input", updateSettings);
  byId("run-history").addEventListener("change", event => selectRun(event.target.value));
  byId("matching-runs").addEventListener("change", renderComparison);
  byId("repeat-settings").addEventListener("click", () => {
    if (!run || activeStates.has(activeRun?.status) || batch?.remaining) return;
    restoreConfig({...run.config, resume: false, resume_run_id: null, seed_count: 1, minimal_layout_dimension: false, automatic_time_steps: false});
    byId("configuration-title").scrollIntoView({block: "start"});
    message("Settings copied. Change the seed or budget if you like, then start a new run.", "info");
  });
  byId("continue-run").addEventListener("click", () => {
    if (!run?.checkpoint_available || activeStates.has(run.status) || activeStates.has(activeRun?.status) || batch?.remaining || pending) return;
    restoreConfig({...run.config, resume: true, resume_run_id: run.id, seed_count: 1, minimal_layout_dimension: false, automatic_time_steps: false});
    byId("configuration-title").scrollIntoView({block: "start"});
    message(`Checkpoint source fixed to run ${run.id.slice(0, 8)}. Set the additional timestep budget, then start when ready. Continuing creates a new run and keeps the original results.`, "info");
  });
  byId("clear-resume-source").addEventListener("click", () => {
    resumeRunId = null;
    updateSettings();
  });
  byId("replay-frame").addEventListener("input", event => { stopReplay(); showFrame(Number(event.target.value)); });
  byId("replay-live").addEventListener("click", () => { stopReplay(); showFrame(null); });
  byId("replay-play").addEventListener("click", () => {
    if (replayTimer !== null) { stopReplay(); return; }
    if (!run || run.replay_count < 2) return;
    if (replayFrame === null || replayFrame >= run.replay_count - 1) showFrame(0);
    byId("replay-play").textContent = "Pause";
    byId("replay-play").setAttribute("aria-label", "Pause saved improvements");
    const advance = () => {
      if (replayFrame === null || replayFrame >= (run?.replay_count || 0) - 1) { stopReplay(); return; }
      if (previewKey === previewId()) showFrame(replayFrame + 1);
      replayTimer = window.setTimeout(advance, 900);
    };
    replayTimer = window.setTimeout(advance, 900);
  });

  // Render metadata, not assumptions about a particular clocking scheme or wire direction.
  const tileSize = 48;
  const phaseColors = ["#f0f4fb", "#e4ebf7", "#d6e0f0", "#c8d5e9"];
  const gateColors = { PI: "#dcf2e7", PO: "#dcecff", INV: "#fce4e6", NOT: "#fce4e6", AND: "#e7e3fa", OR: "#f7edce", XOR: "#dbedf2", MAJ: "#e7e3fa", BUF: "#f9f4df" };

  function drawSoon() {
    if (drawFrame !== null) return;
    drawFrame = requestAnimationFrame(() => { drawFrame = null; draw(); });
  }

  function fit() {
    if (!preview) return;
    const bounds = byId("canvas-wrap").getBoundingClientRect();
    view.scale = Math.min((bounds.width - 64) / (preview.width * tileSize), (bounds.height - 80) / (preview.height * tileSize), document.fullscreenElement ? 5 : 2);
    view.scale = Math.max(.015, view.scale);
    view.x = (bounds.width - preview.width * tileSize * view.scale) / 2;
    view.y = (bounds.height - preview.height * tileSize * view.scale) / 2;
    drawSoon();
  }

  function cellCenter(coordinate) {
    const [x, y] = coordinate;
    return { x: x * tileSize + tileSize / 2, y: y * tileSize + tileSize / 2 };
  }

  function draw(output = canvas, viewport = view, bounds = byId("canvas-wrap").getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2), exporting = false) {
    const canvas = output, context = output.getContext("2d"), view = viewport;
    if (!preview || !context) return;
    if (canvas.width !== Math.round(bounds.width * ratio) || canvas.height !== Math.round(bounds.height * ratio)) {
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    if (exporting) {
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, bounds.width, bounds.height);
    }
    context.translate(view.x, view.y);
    context.scale(view.scale, view.scale);
    const small = tileSize * view.scale < 25;
    const startX = Math.max(0, Math.floor(-view.x / view.scale / tileSize));
    const startY = Math.max(0, Math.floor(-view.y / view.scale / tileSize));
    const endX = Math.min(preview.width, Math.ceil((bounds.width - view.x) / view.scale / tileSize));
    const endY = Math.min(preview.height, Math.ceil((bounds.height - view.y) / view.scale / tileSize));
    for (let y = startY; y < endY; y++) {
      for (let x = startX; x < endX; x++) {
        const phase = preview.phases?.[y]?.[x] ?? cellsByTile.get(`${x},${y}`)?.[0]?.phase;
        context.fillStyle = phaseColors[(Number(phase) - 1) % 4] || "#e9edf5";
        context.fillRect(x * tileSize + 1, y * tileSize + 1, tileSize - 2, tileSize - 2);
        if (!small && phase != null) {
          context.fillStyle = "#7888a3";
          context.font = "8px -apple-system, BlinkMacSystemFont, sans-serif";
          context.textAlign = "right";
          context.fillText(String(phase), (x + 1) * tileSize - 5, (y + 1) * tileSize - 5);
        }
      }
    }
    // Keep both wires on-grid: solid ground first, dashed upper layer second.
    for (const upper of [false, true]) {
      for (const edge of preview.edges || []) {
        if ((edge.source[2] > 0 || edge.target[2] > 0) !== upper) continue;
        if (Math.max(edge.source[0], edge.target[0]) < startX || Math.min(edge.source[0], edge.target[0]) >= endX || Math.max(edge.source[1], edge.target[1]) < startY || Math.min(edge.source[1], edge.target[1]) >= endY) continue;
        const start = cellCenter(edge.source);
        const end = cellCenter(edge.target);
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        const targetX = end.x - Math.cos(angle) * 8;
        const targetY = end.y - Math.sin(angle) * 8;
        context.strokeStyle = !exporting && changedEdges.has(JSON.stringify(edge)) ? "#ce7628" : upper ? "#9673c6" : "#5382c8";
        context.fillStyle = context.strokeStyle;
        context.lineWidth = small ? 2.4 : 1.8;
        context.setLineDash(upper ? [4, 3] : []);
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
        context.setLineDash([]);
        if (!small) {
          context.beginPath();
          context.moveTo(targetX, targetY);
          context.lineTo(targetX - Math.cos(angle - .5) * 5, targetY - Math.sin(angle - .5) * 5);
          context.lineTo(targetX - Math.cos(angle + .5) * 5, targetY - Math.sin(angle + .5) * 5);
          context.closePath();
          context.fill();
        }
      }
    }
    for (const cell of preview.cells) {
      if (cell.x < startX || cell.x >= endX || cell.y < startY || cell.y >= endY) continue;
      const center = cellCenter([cell.x, cell.y, cell.z]);
      const type = String(cell.type).toUpperCase();
      context.fillStyle = gateColors[type] || "#e7e3f6";
      context.strokeStyle = cell.z > 0 ? "#9673c6" : "#8fa2be";
      context.lineWidth = 1;
      if (!exporting && changedCells.has(`${cell.x},${cell.y},${cell.z}`)) {
        context.strokeStyle = "#ce7628";
        context.lineWidth = 2;
        context.strokeRect(cell.x * tileSize + 5, cell.y * tileSize + 5, tileSize - 10, tileSize - 10);
        context.lineWidth = 1;
      }
      if (type === "BUF") {
        const layers = cellsByTile.get(`${cell.x},${cell.y}`);
        if (layers.length > 1) {
          // One neutral tile marker; the separate edges still retain their native layers.
          if (cell !== layers[0]) continue;
          context.fillStyle = "#fff";
          context.strokeStyle = "#8fa2be";
        }
        context.beginPath();
        context.arc(center.x, center.y, 5, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      } else {
        context.fillRect(center.x - 12, center.y - 8, 24, 16);
        context.strokeRect(center.x - 12, center.y - 8, 24, 16);
        if (!small) {
          context.fillStyle = "#485878";
          context.textAlign = "center";
          context.font = "7px -apple-system, BlinkMacSystemFont, sans-serif";
          context.fillText(type, center.x, center.y + 2.5, 23);
        }
      }
    }
    if (selectedTile && !exporting) {
      context.strokeStyle = "#246af3";
      context.lineWidth = 2 / view.scale;
      context.strokeRect(selectedTile.x * tileSize, selectedTile.y * tileSize, tileSize, tileSize);
    }
  }

  function zoom(factor, x, y) {
    if (!preview) return;
    const bounds = byId("canvas-wrap").getBoundingClientRect();
    x ??= bounds.width / 2;
    y ??= bounds.height / 2;
    const next = Math.max(.015, Math.min(6, view.scale * factor));
    view.x = x - (x - view.x) * next / view.scale;
    view.y = y - (y - view.y) * next / view.scale;
    view.scale = next;
    drawSoon();
  }

  function inspect(x, y) {
    if (!preview) return;
    const column = Math.floor((x - view.x) / view.scale / tileSize);
    const row = Math.floor((y - view.y) / view.scale / tileSize);
    if (column < 0 || row < 0 || column >= preview.width || row >= preview.height) return;
    selectedTile = { x: column, y: row };
    const cells = cellsByTile.get(`${column},${row}`) || [];
    const phase = preview.phases?.[row]?.[column] ?? cells[0]?.phase;
    const detail = cells.length ? cells.map((cell) => `${cell.type}${cell.name ? ` “${cell.name}”` : ""} · layer ${cell.z}`).join(" / ") : "Empty tile";
    byId("canvas-detail").textContent = `(${column}, ${row}) · phase ${phase ?? "unknown"} · ${detail}`;
    drawSoon();
  }

  byId("zoom-in").addEventListener("click", () => zoom(1.3));
  byId("zoom-out").addEventListener("click", () => zoom(1 / 1.3));
  byId("fit-view").addEventListener("click", fit);
  byId("fullscreen-view").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await byId("preview-card").requestFullscreen();
    } catch (error) { message(`Fullscreen is unavailable. ${error.message}`); }
  });
  document.addEventListener("fullscreenchange", () => {
    byId("fullscreen-view").setAttribute("aria-label", document.fullscreenElement ? "Exit fullscreen" : "Fullscreen layout");
    fit();
  });
  byId("export-png").addEventListener("click", () => {
    if (!preview || previewKey !== previewId()) return;
    const output = document.createElement("canvas");
    const bounds = {width: preview.width * tileSize + 48, height: preview.height * tileSize + 48};
    const ratio = 4096 / Math.max(bounds.width, bounds.height);
    draw(output, {x: 24, y: 24, scale: 1}, bounds, ratio, true);
    const name = `${run?.config.function || "layout"}-${replayFrame === null ? "best" : `improvement-${replayFrame + 1}`}.png`;
    output.toBlob(blob => {
      if (!blob) { message("The browser could not export this layout."); return; }
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/png");
  });
  canvas.addEventListener("wheel", (event) => {
    if (!preview) return;
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    zoom(Math.exp(-event.deltaY * .0015), event.clientX - bounds.left, event.clientY - bounds.top);
  }, { passive: false });
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: 0 };
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!pointer || pointer.id !== event.pointerId) return;
    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    view.x += dx;
    view.y += dy;
    pointer.moved += Math.abs(dx) + Math.abs(dy);
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    drawSoon();
  });
  canvas.addEventListener("pointerup", (event) => {
    if (pointer?.id === event.pointerId && pointer.moved < 5) {
      const bounds = canvas.getBoundingClientRect();
      inspect(event.clientX - bounds.left, event.clientY - bounds.top);
    }
    pointer = null;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("pointercancel", () => { pointer = null; canvas.classList.remove("dragging"); });
  canvas.addEventListener("keydown", (event) => {
    if (!preview) return;
    if (event.key === "+" || event.key === "=") zoom(1.3);
    else if (event.key === "-") zoom(1 / 1.3);
    else if (event.key === "0" || event.key === "Home") fit();
    else if (event.key === "ArrowLeft") view.x += 40;
    else if (event.key === "ArrowRight") view.x -= 40;
    else if (event.key === "ArrowUp") view.y += 40;
    else if (event.key === "ArrowDown") view.y -= 40;
    else if (event.key === "Enter") {
      const bounds = canvas.getBoundingClientRect();
      inspect(bounds.width / 2, bounds.height / 2);
    } else return;
    event.preventDefault();
    drawSoon();
  });
  new ResizeObserver(() => { if (preview) fit(); }).observe(byId("canvas-wrap"));

  async function initialize() {
    try {
      catalog = await api("/api/catalog");
      fillSelect(benchmark, Object.keys(catalog.benchmarks), "trindade16");
      fillCircuits();
      fillSelect(technology, catalog.technologies, "Gate-level");
      fillSelect(clocking, catalog.clocking_schemes, "2DDWave");
      width.max = height.max = catalog.limits.max_dimension;
      steps.max = catalog.limits.max_timesteps;
      byId("version").textContent = catalog.version ? `v${catalog.version}` : "";
      updateSettings();
      message("");
      poll();
    } catch (error) {
      message(`Could not load the local workspace. Retrying automatically. ${error.message}`);
      window.setTimeout(initialize, 3000);
    }
  }
  initialize();
})();
