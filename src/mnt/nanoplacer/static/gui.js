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
  const seed = byId("seed");
  const optimize = byId("optimize");
  const resume = byId("resume");
  const canvas = byId("layout-canvas");
  const context = canvas.getContext("2d");
  const token = document.querySelector('meta[name="csrf-token"]').content;
  const activeStates = new Set(["queued", "starting", "running", "cancelling"]);
  const numbers = new Intl.NumberFormat();
  let catalog = null;
  let run = null;
  let seenRunId = null;
  let canResume = false;
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

  function updateSettings() {
    if (!catalog) return;
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
    height.setCustomValidity(area > catalog.limits.max_tiles ? `Use at most ${numbers.format(catalog.limits.max_tiles)} tiles.` : "");
    resume.disabled = !canResume;
    if (!canResume) resume.checked = false;
    byId("resume-hint").textContent = canResume
      ? "A saved model is available. The server will verify that its settings match."
      : "Available after a model has been saved. Settings must match.";
    updateDisabled();
  }

  function updateDisabled() {
    const active = Boolean(run && activeStates.has(run.status));
    fields.disabled = !catalog || !statusReady || pending || active;
    byId("start-button").disabled = !catalog || !statusReady || pending || active;
    byId("start-button").textContent = pending ? "Starting…" : active ? "Experiment in progress" : "Start experiment";
    byId("cancel-button").hidden = !active;
    byId("cancel-button").disabled = cancelling || run?.status === "cancelling";
    byId("cancel-button").textContent = cancelling || run?.status === "cancelling" ? "Stopping…" : "Stop run";
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
    seed.value = config.seed ?? 42;
    optimize.checked = Boolean(config.optimize);
    resume.checked = Boolean(config.resume);
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
      seed: Number(seed.value),
      optimize: optimize.checked,
      minimal_layout_dimension: minimum.checked,
      resume: resume.checked,
    };
  }

  const count = (value) => value == null ? "—" : numbers.format(value);
  function elapsed(value) {
    if (value == null) return "—";
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
    preview = null;
    previewKey = null;
    selectedTile = null;
    cellsByTile = new Map();
    canvas.hidden = true;
    byId("preview-empty").hidden = false;
    byId("preview-kind").hidden = true;
    byId("preview-title").textContent = "A place for every gate";
    byId("preview-dimensions").textContent = "—";
    byId("canvas-detail").textContent = "Gate-level preview · read-only";
    for (const id of ["zoom-in", "zoom-out", "fit-view"]) byId(id).disabled = true;
  }

  function renderStatus(data) {
    statusReady = true;
    run = data.run;
    drawReward();
    canResume = Boolean(data.can_resume);
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
        for (const id of ["steps-value", "episodes-value", "elapsed-value", "placed-value"]) byId(id).textContent = "—";
        byId("run-outcome").textContent = "No experiment is active in this server session.";
        byId("run-log").textContent = "Logs will appear after a run starts.";
        renderFiles([]);
      }
      updateSettings();
      return;
    }
    if (run.id !== seenRunId) {
      seenRunId = run.id;
      clearPreview();
      restoreConfig(run.config);
    }
    updateSettings();
    const active = activeStates.has(run.status);
    const status = byId("run-status");
    status.textContent = run.status;
    status.dataset.state = run.status;
    const titles = { starting: "Preparing the agent", queued: "Preparing the agent", running: "Exploring the possibilities", cancelling: "Finishing this run", cancelled: "Experiment stopped", completed: "Experiment complete", failed: "Experiment needs attention" };
    byId("run-title").textContent = titles[run.status] || "Your experiment";
    byId("run-description").textContent = `${run.config.benchmark} / ${run.config.function} · ${run.config.clocking_scheme} · ${run.config.technology}`;
    const actual = Number(run.timesteps) || 0;
    const total = Number(run.total_timesteps ?? run.config.time_steps) || 1;
    const percent = Math.min(100, Math.max(0, 100 * actual / total));
    byId("training-progress").value = percent;
    byId("progress-value").textContent = `${Math.floor(percent)}%`;
    byId("steps-value").textContent = count(run.timesteps);
    byId("episodes-value").textContent = count(run.episodes);
    byId("elapsed-value").textContent = elapsed(run.elapsed);
    byId("placed-value").textContent = run.total_nodes == null ? count(run.best_placed) : `${count(run.best_placed)} / ${count(run.total_nodes)}`;
    byId("progress-label").textContent = active && actual > total ? "Budget reached · finishing rollout" : `Budget · ${count(total)} timesteps`;
    let outcome;
    if (run.error) outcome = run.error;
    else if (run.solution_found) {
      const equivalence = {
        STRONG: "Complete layout found · strong equivalence verified.",
        WEAK: "Complete layout found · weak equivalence (timing differs).",
        NO: "Complete layout found · equivalence check failed. Review the results.",
      };
      outcome = equivalence[run.equivalent] || "Complete layout found · equivalence not verified.";
    }
    else if (run.status === "cancelled") outcome = "Stopped. Available partial results and saved files are preserved.";
    else if (run.status === "completed") outcome = "Budget finished without a complete layout. Try another seed, more timesteps, or a larger grid.";
    else outcome = "Searching for a complete placement. The preview shows the best result observed so far.";
    if (byId("run-outcome").textContent !== outcome) byId("run-outcome").textContent = outcome;
    byId("preview-empty-description").textContent = active
      ? "Waiting for the first observed placement. This animation is an illustration, not a training result."
      : "No placement preview is available for this run. Check the training log for details.";
    byId("run-log").textContent = run.log || "No log output yet.";
    renderFiles(run.files || []);
    updatePreviewLabel();
    const key = `${run.id}:${run.preview_revision}`;
    if (run.preview_revision != null && run.preview_revision > 0 && key !== previewKey && !previewRequest) {
      previewRequest = key;
      api(`/api/preview?revision=${encodeURIComponent(run.preview_revision)}`).then((data) => {
        if (!run || `${run.id}:${run.preview_revision}` !== key) return;
        if (!data || !data.width || !data.height || !Array.isArray(data.cells)) return;
        const refit = !preview || preview.width !== data.width || preview.height !== data.height;
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
        byId("preview-title").textContent = "The best so far";
        byId("preview-dimensions").textContent = `${data.width} × ${data.height} · ${data.clocking_scheme}`;
        for (const id of ["zoom-in", "zoom-out", "fit-view"]) byId(id).disabled = false;
        updatePreviewLabel();
        if (refit) fit(); else drawSoon();
      }).catch((error) => {
        byId("canvas-detail").textContent = `Preview temporarily unavailable: ${error.message}`;
      }).finally(() => { previewRequest = null; });
    }
  }

  function updatePreviewLabel() {
    const complete = Boolean(run?.solution_found && previewKey === `${run.id}:${run.preview_revision}`);
    byId("preview-kind").textContent = complete ? "Best complete layout" : "Best partial placement";
    byId("preview-kind").dataset.complete = String(complete);
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
    const epoch = statusEpoch;
    try {
      const data = await api("/api/status");
      // A status request sent before Start/Stop must not overwrite its response.
      if (epoch !== statusEpoch || pending || cancelling) return;
      renderStatus(data);
      if (connectionError) {
        connectionError = false;
        message("");
      }
    } catch (error) {
      connectionError = true;
      message(`Connection interrupted. Retrying automatically. ${error.message}`);
    } finally {
      window.setTimeout(poll, 750);
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!catalog || !statusReady || pending || activeStates.has(run?.status)) return;
    updateSettings();
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
      renderStatus(await api("/api/start", config()));
    } catch (error) {
      message(`Could not start the experiment. ${error.message}`);
    } finally {
      pending = false;
      statusEpoch += 1;
      updateDisabled();
    }
  });

  byId("cancel-button").addEventListener("click", async () => {
    if (cancelling || !activeStates.has(run?.status)) return;
    cancelling = true;
    statusEpoch += 1;
    updateDisabled();
    try {
      renderStatus(await api("/api/cancel", {}));
    } catch (error) {
      message(`Could not stop the experiment. ${error.message}`);
    } finally {
      cancelling = false;
      statusEpoch += 1;
      updateDisabled();
    }
  });

  benchmark.addEventListener("change", () => { fillCircuits(); updateSettings(); });
  for (const element of [circuit, clocking, technology, minimum]) element.addEventListener("change", updateSettings);
  for (const element of [width, height]) element.addEventListener("input", updateSettings);

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
    view.scale = Math.min((bounds.width - 64) / (preview.width * tileSize), (bounds.height - 80) / (preview.height * tileSize), 2);
    view.scale = Math.max(.015, view.scale);
    view.x = (bounds.width - preview.width * tileSize * view.scale) / 2;
    view.y = (bounds.height - preview.height * tileSize * view.scale) / 2;
    drawSoon();
  }

  function cellCenter(coordinate) {
    const [x, y] = coordinate;
    return { x: x * tileSize + tileSize / 2, y: y * tileSize + tileSize / 2 };
  }

  function draw() {
    if (!preview || !context) return;
    const bounds = byId("canvas-wrap").getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(bounds.width * ratio) || canvas.height !== Math.round(bounds.height * ratio)) {
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
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
        const start = cellCenter(edge.source);
        const end = cellCenter(edge.target);
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        const targetX = end.x - Math.cos(angle) * 8;
        const targetY = end.y - Math.sin(angle) * 8;
        context.strokeStyle = upper ? "#9673c6" : "#5382c8";
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
    if (selectedTile) {
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
