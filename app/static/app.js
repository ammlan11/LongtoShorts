// ---- element refs -----------------------------------------------------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const selectedFileBox = document.getElementById("selected-file");
const selectedFileName = document.getElementById("selected-file-name");
const clearFileBtn = document.getElementById("clear-file");
const startBtn = document.getElementById("start-btn");

const tabFile = document.getElementById("tab-file");
const tabUrl = document.getElementById("tab-url");
const fileSource = document.getElementById("file-source");
const urlSource = document.getElementById("url-source");
const urlInput = document.getElementById("url-input");

const uploadPanel = document.getElementById("upload-panel");
const progressPanel = document.getElementById("progress-panel");
const progressStage = document.getElementById("progress-stage");
const progressFill = document.getElementById("progress-fill");
const progressNote = document.getElementById("progress-note");

const reviewPanel = document.getElementById("review-panel");
const reviewSubtitle = document.getElementById("review-subtitle");
const candidatesList = document.getElementById("candidates-list");
const captionPresetSelect = document.getElementById("caption-preset-select");
const aspectPresetSelect = document.getElementById("aspect-preset-select");
const captionPresetDescription = document.getElementById("caption-preset-description");
const customCaptionControls = document.getElementById("custom-caption-controls");
const customPosition = document.getElementById("custom-position");
const customFontSize = document.getElementById("custom-font-size");
const customFontSizeValue = document.getElementById("custom-font-size-value");
const customTextColor = document.getElementById("custom-text-color");
const customHighlightColor = document.getElementById("custom-highlight-color");
const customBackgroundBox = document.getElementById("custom-background-box");
const renderBtn = document.getElementById("render-btn");
const backToUploadBtn = document.getElementById("back-to-upload-btn");

const resultsPanel = document.getElementById("results-panel");
const resultsGrid = document.getElementById("results-grid");
const startOverBtn = document.getElementById("start-over-btn");

const stepEls = {
  1: document.getElementById("step-1"),
  2: document.getElementById("step-2"),
  3: document.getElementById("step-3"),
};

let selectedFile = null;
let sourceMode = "file"; // "file" | "url"
let currentJobId = null;
let pollStartedAt = null;
let candidates = []; // working copies of the candidates returned by analyze, keyed by id
let captionPresets = [];
let currentPollTimer = null;

const STAGE_LABELS = {
  downloading: "Downloading video…",
  transcribing: "Transcribing audio (local Whisper)…",
  finding_highlights: "Finding candidate moments…",
  rendering: "Reframing, adding captions & graphics…",
  done: "Done",
};

function setStep(n) {
  Object.entries(stepEls).forEach(([k, el]) => el.classList.toggle("active", Number(k) <= n));
}

// ---- step 1: upload -----------------------------------------------------

function setSourceMode(mode) {
  sourceMode = mode;
  tabFile.classList.toggle("active", mode === "file");
  tabUrl.classList.toggle("active", mode === "url");
  fileSource.hidden = mode !== "file";
  urlSource.hidden = mode !== "url";
  updateStartEnabled();
}

function updateStartEnabled() {
  startBtn.disabled = sourceMode === "file" ? !selectedFile : !urlInput.value.trim();
}

tabFile.addEventListener("click", () => setSourceMode("file"));
tabUrl.addEventListener("click", () => setSourceMode("url"));
urlInput.addEventListener("input", updateStartEnabled);

dropzone.addEventListener("click", () => fileInput.click());
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
clearFileBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  selectedFileBox.hidden = true;
  dropzone.hidden = false;
  updateStartEnabled();
});

function setFile(file) {
  selectedFile = file;
  selectedFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  selectedFileBox.hidden = false;
  dropzone.hidden = true;
  updateStartEnabled();
}

startBtn.addEventListener("click", async () => {
  if (sourceMode === "file" && !selectedFile) return;
  if (sourceMode === "url" && !urlInput.value.trim()) return;

  startBtn.disabled = true;
  showOnly(progressPanel);
  setStep(1);
  progressFill.style.width = "2%";
  progressStage.textContent = sourceMode === "url" ? "Starting download…" : "Uploading…";
  progressNote.textContent = "";

  const form = new FormData();
  form.append("clip_count", document.getElementById("clip-count").value);
  form.append("captions_enabled", document.getElementById("captions-enabled").checked);
  form.append("use_llm", document.getElementById("use-llm").checked);

  const endpoint = sourceMode === "url" ? "/api/upload-url" : "/api/upload";
  if (sourceMode === "url") {
    form.append("url", urlInput.value.trim());
  } else {
    form.append("file", selectedFile);
  }

  try {
    const res = await fetch(endpoint, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error((body && body.detail) || `Request failed (${res.status})`);
    }
    const { job_id } = await res.json();
    currentJobId = job_id;
    pollUntil(job_id, "awaiting_review", onAnalyzeDone);
  } catch (err) {
    showError(err.message);
  }
});

// ---- polling --------------------------------------------------------------

// Polls /api/jobs/{id} until status becomes `targetStatus` (or "done"/"error",
// which always stop polling), updating the shared progress panel as it goes.
function pollUntil(jobId, targetStatus, onReached) {
  if (currentPollTimer) clearInterval(currentPollTimer);
  pollStartedAt = Date.now();
  currentPollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();

      if (job.status === "error") {
        clearInterval(currentPollTimer);
        showError(job.error || "Something went wrong.");
        return;
      }

      const stageLabel = STAGE_LABELS[job.stage] || job.stage || "Working…";
      progressStage.textContent = stageLabel;
      progressFill.style.width = `${Math.max(2, Math.round(job.progress * 100))}%`;
      const elapsedSec = (Date.now() - pollStartedAt) / 1000;
      progressNote.textContent = `Elapsed: ${formatElapsed(elapsedSec)} — ${Math.round(job.progress * 100)}% through this stage`;

      if (job.status === targetStatus || job.status === "done") {
        clearInterval(currentPollTimer);
        onReached(job);
      }
    } catch (err) {
      clearInterval(currentPollTimer);
      showError(err.message);
    }
  }, 1500);
}

// ---- step 2: review & style -----------------------------------------------

async function loadPresetsOnce() {
  if (captionPresets.length) return;
  const res = await fetch("/api/presets");
  const data = await res.json();
  captionPresets = data.captions;

  captionPresetSelect.innerHTML = data.captions
    .map((p) => `<option value="${p.id}" ${p.id === data.default_caption_preset ? "selected" : ""}>${escapeHtml(p.label)}</option>`)
    .join("") + `<option value="custom">Custom…</option>`;
  aspectPresetSelect.innerHTML = data.aspects
    .map((p) => `<option value="${p.id}" ${p.id === data.default_aspect_preset ? "selected" : ""}>${escapeHtml(p.label)}</option>`)
    .join("");

  updateCaptionDescription();
  captionPresetSelect.addEventListener("change", updateCaptionDescription);
  customFontSize.addEventListener("input", () => {
    customFontSizeValue.textContent = `${customFontSize.value}px`;
  });
}

function updateCaptionDescription() {
  const isCustom = captionPresetSelect.value === "custom";
  customCaptionControls.hidden = !isCustom;
  if (isCustom) {
    captionPresetDescription.textContent = "Set your own position, size, and colors below.";
    return;
  }
  const preset = captionPresets.find((p) => p.id === captionPresetSelect.value);
  captionPresetDescription.textContent = preset ? preset.description : "";
}

async function onAnalyzeDone(job) {
  await loadPresetsOnce();

  // Default a likely recap/preview clip to unchecked -- it's still shown
  // (and still scoreable/pickable) but shouldn't get rendered without the
  // user actively choosing it, since it can spoil/duplicate the rest of
  // the video.
  candidates = (job.candidates || []).map((c) => ({ ...c, selected: !c.is_recap }));

  if (candidates.length === 0) {
    showOnly(reviewPanel);
    setStep(2);
    reviewSubtitle.textContent = "";
    candidatesList.innerHTML = `<p class="note">No highlight-worthy moments were found in this video.</p>`;
    renderBtn.hidden = true;
    return;
  }
  renderBtn.hidden = false;

  const timingNote = formatTimingBreakdown(job.timing);
  reviewSubtitle.textContent = `Found ${candidates.length} possible shorts (source: ${formatDuration(job.source_duration)}). Uncheck any you don't want, nudge the start/end if needed, then pick a look and render.${timingNote ? " " + timingNote : ""}`;
  renderCandidatesList();
  showOnly(reviewPanel);
  setStep(2);
}

function renderCandidatesList() {
  candidatesList.innerHTML = "";
  candidates.forEach((c) => {
    const card = document.createElement("div");
    card.className = "candidate-card";
    const reasons = (c.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");
    card.innerHTML = `
      <label class="candidate-select">
        <input type="checkbox" data-role="selected" ${c.selected ? "checked" : ""} />
      </label>
      <div class="candidate-body">
        ${c.most_replayed ? `<div class="most-replayed-badge">🔥 Most replayed (YouTube data)</div>` : ""}
        ${c.is_recap ? `<div class="recap-badge">⚠️ Looks like a recap/preview — unchecked by default</div>` : ""}
        <input class="candidate-title" data-role="title" type="text" value="${escapeAttr(c.title)}" />
        <div class="candidate-times">
          <label>Start (s) <input data-role="start" type="number" min="0" step="0.5" value="${c.start}" /></label>
          <label>End (s) <input data-role="end" type="number" min="0" step="0.5" value="${c.end}" /></label>
          <span class="candidate-duration" data-role="duration">${(c.end - c.start).toFixed(1)}s</span>
          <span class="candidate-score">score ${c.score}</span>
        </div>
        <ul class="candidate-reasons">${reasons}</ul>
      </div>
    `;

    const checkbox = card.querySelector('[data-role="selected"]');
    const titleInput = card.querySelector('[data-role="title"]');
    const startInput = card.querySelector('[data-role="start"]');
    const endInput = card.querySelector('[data-role="end"]');
    const durationEl = card.querySelector('[data-role="duration"]');

    checkbox.addEventListener("change", () => {
      c.selected = checkbox.checked;
      card.classList.toggle("deselected", !c.selected);
    });
    titleInput.addEventListener("input", () => { c.title = titleInput.value; });
    startInput.addEventListener("input", () => {
      c.start = parseFloat(startInput.value) || 0;
      durationEl.textContent = `${Math.max(0, c.end - c.start).toFixed(1)}s`;
    });
    endInput.addEventListener("input", () => {
      c.end = parseFloat(endInput.value) || 0;
      durationEl.textContent = `${Math.max(0, c.end - c.start).toFixed(1)}s`;
    });

    card.classList.toggle("deselected", !c.selected);
    candidatesList.appendChild(card);
  });
}

backToUploadBtn.addEventListener("click", () => {
  resetToUpload();
});

renderBtn.addEventListener("click", async () => {
  const selections = candidates
    .filter((c) => c.selected && c.end > c.start)
    .map((c) => ({ id: c.id, start: c.start, end: c.end, title: c.title, score: c.score, reasons: c.reasons, most_replayed: c.most_replayed, is_recap: c.is_recap }));

  if (selections.length === 0) {
    window.alert("Pick at least one short to render.");
    return;
  }

  renderBtn.disabled = true;
  showOnly(progressPanel);
  setStep(2);
  progressFill.style.width = "2%";
  progressStage.textContent = "Starting render…";
  progressNote.textContent = "";

  const isCustom = captionPresetSelect.value === "custom";
  const payload = {
    selections,
    caption_preset: captionPresetSelect.value,
    aspect_preset: aspectPresetSelect.value,
  };
  if (isCustom) {
    payload.caption_overrides = {
      position: customPosition.value,
      font_size: parseInt(customFontSize.value, 10),
      text_color: customTextColor.value,
      highlight_color: customHighlightColor.value,
      background_box: customBackgroundBox.checked,
    };
  }

  try {
    const res = await fetch(`/api/jobs/${currentJobId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error((body && body.detail) || `Request failed (${res.status})`);
    }
    pollUntil(currentJobId, "done", (job) => renderResults(currentJobId, job.summary));
  } catch (err) {
    showError(err.message);
  } finally {
    renderBtn.disabled = false;
  }
});

// ---- step 3: results --------------------------------------------------------

function renderResults(jobId, summary) {
  showOnly(resultsPanel);
  setStep(3);

  if (!summary || !summary.shorts || summary.shorts.length === 0) {
    resultsGrid.innerHTML = `<p class="note">Nothing was rendered.</p>`;
    return;
  }

  resultsGrid.innerHTML = "";
  summary.shorts.forEach((short) => {
    resultsGrid.appendChild(buildResultCard(jobId, short));
  });
}

startOverBtn.addEventListener("click", () => resetToUpload());

// ---- step 3b: results-page editing (trim / tags / emphasis / AI captions) --

function metaText(short) {
  return `${short.duration.toFixed(0)}s · score ${short.score}${short.overlay_count ? ` · ${short.overlay_count} graphic overlay(s)` : ""}`;
}

function tagContainerHtml(short) {
  const tags = (short.tags || []).map((t) => `<span class="tag-pill">#${escapeHtml(t)}</span>`).join("");
  const emphasis = (short.emphasis_words || [])
    .map((w) => `<span class="tag-pill emphasis-pill">✨ ${escapeHtml(w)}</span>`)
    .join("");
  if (!tags && !emphasis) return "";
  return `${tags ? `<div class="tag-row">${tags}</div>` : ""}${emphasis ? `<div class="tag-row">${emphasis}</div>` : ""}`;
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error((body && body.detail) || `Request failed (${res.status})`);
  }
  return res.json();
}

function buildResultCard(jobId, short) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.dataset.index = short.index;

  const downloadUrl = `/api/jobs/${jobId}/download/${short.file}`;
  const reasons = (short.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");

  card.innerHTML = `
    <video src="${downloadUrl}" controls preload="metadata" data-role="video"></video>
    <div class="result-body">
      ${short.most_replayed ? `<div class="most-replayed-badge">🔥 Most replayed (YouTube data)</div>` : ""}
      ${short.is_recap ? `<div class="recap-badge">⚠️ Recap/preview clip</div>` : ""}
      <div class="result-title" data-role="title-display">${escapeHtml(short.title || "Untitled short")}</div>
      <ul class="result-reasons">${reasons}</ul>
      <div class="result-meta" data-role="meta-display">${metaText(short)}</div>
      <div class="tag-container" data-role="tag-container">${tagContainerHtml(short)}</div>
      <div class="result-btn-row">
        <a class="result-download" href="${downloadUrl}" download data-role="download-link">Download</a>
        <button class="secondary-btn edit-toggle-btn" type="button">Edit</button>
      </div>
      <div class="result-edit" hidden></div>
    </div>
  `;

  const editToggleBtn = card.querySelector(".edit-toggle-btn");
  const editPanel = card.querySelector(".result-edit");
  editToggleBtn.addEventListener("click", () => {
    const willOpen = editPanel.hidden;
    editPanel.hidden = !willOpen;
    editToggleBtn.textContent = willOpen ? "Close edit" : "Edit";
    if (willOpen && !editPanel.dataset.built) {
      buildEditPanel(editPanel, card, jobId, short);
      editPanel.dataset.built = "1";
    }
  });

  return card;
}

// Re-renders the bits of a result card that can change after a post-render
// edit (title, meta line, tag/emphasis pills, and the video itself -- which
// needs a cache-busting query param since trim/emphasis rewrite the same
// filename in place).
function refreshResultCard(card, jobId, short, { videoChanged = false } = {}) {
  card.querySelector('[data-role="title-display"]').textContent = short.title || "Untitled short";
  card.querySelector('[data-role="meta-display"]').textContent = metaText(short);
  card.querySelector('[data-role="tag-container"]').innerHTML = tagContainerHtml(short);
  if (videoChanged) {
    const bust = Date.now();
    card.querySelector('[data-role="video"]').src = `/api/jobs/${jobId}/download/${short.file}?v=${bust}`;
    card.querySelector('[data-role="download-link"]').href = `/api/jobs/${jobId}/download/${short.file}?v=${bust}`;
  }
}

function buildEditPanel(editPanel, card, jobId, short) {
  editPanel.innerHTML = `
    <div class="edit-section">
      <h4>Trim</h4>
      <p class="note">Cuts the already-rendered file directly, so it's fast — but captions or overlays near the new edge may end abruptly rather than re-timing to it.</p>
      <div class="edit-row">
        <label>Trim from start (s) <input type="number" min="0" step="0.5" value="0" data-role="trim-start" /></label>
        <label>Trim from end (s) <input type="number" min="0" step="0.5" value="0" data-role="trim-end" /></label>
        <button class="edit-apply-btn" data-role="trim-apply" type="button">Apply trim</button>
      </div>
    </div>

    <div class="edit-section">
      <h4>Title &amp; keywords</h4>
      <p class="note">Keywords are saved as upload-ready hashtags alongside the clip (a .txt file next to the download).</p>
      <div class="edit-row">
        <label class="edit-row-full">Title
          <input type="text" data-role="title-input" value="${escapeAttr(short.title || "")}" />
        </label>
        <label class="edit-row-full">Keywords / hashtags (comma-separated)
          <input type="text" placeholder="e.g. trading, mindset, motivation" data-role="tags-input" value="${escapeAttr((short.tags || []).join(", "))}" />
        </label>
        <button class="edit-apply-btn" data-role="save-details" type="button">Save title &amp; keywords</button>
      </div>
    </div>

    <div class="edit-section">
      <h4>Emphasize keywords in captions</h4>
      <p class="note">These words get the highlighted caption treatment every time they're spoken in this clip. Re-burns the captions, so it takes a few seconds.</p>
      <div class="edit-row">
        <label class="edit-row-full">Words to emphasize (comma-separated)
          <input type="text" placeholder="e.g. profit, secret" data-role="emphasis-input" value="${escapeAttr((short.emphasis_words || []).join(", "))}" />
        </label>
        <button class="edit-apply-btn" data-role="emphasis-apply" type="button">Re-burn captions</button>
      </div>
    </div>

    <div class="edit-section">
      <h4>✨ AI caption suggestions</h4>
      <p class="note">Uses your local LLM (Ollama) if it's running; otherwise falls back to a quick built-in heuristic — either way, nothing leaves this machine.</p>
      <button class="edit-apply-btn" data-role="suggest-btn" type="button">Suggest captions</button>
      <div class="suggestions-list" data-role="suggestions"></div>
    </div>

    <div class="edit-status" data-role="edit-status"></div>
  `;

  const statusEl = editPanel.querySelector('[data-role="edit-status"]');
  let statusTimer = null;
  function setStatus(msg, isError) {
    clearTimeout(statusTimer);
    statusEl.textContent = msg;
    statusEl.classList.toggle("edit-status-error", !!isError);
    if (msg && !isError) {
      statusTimer = setTimeout(() => { statusEl.textContent = ""; }, 4000);
    }
  }

  editPanel.querySelector('[data-role="trim-apply"]').addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const trimStart = parseFloat(editPanel.querySelector('[data-role="trim-start"]').value) || 0;
    const trimEnd = parseFloat(editPanel.querySelector('[data-role="trim-end"]').value) || 0;
    if (trimStart <= 0 && trimEnd <= 0) {
      setStatus("Set a trim amount first.", true);
      return;
    }
    btn.disabled = true;
    setStatus("Trimming…");
    try {
      const meta = await postJson(`/api/jobs/${jobId}/shorts/${short.index}/trim`, { trim_start: trimStart, trim_end: trimEnd });
      Object.assign(short, meta);
      refreshResultCard(card, jobId, short, { videoChanged: true });
      editPanel.querySelector('[data-role="trim-start"]').value = 0;
      editPanel.querySelector('[data-role="trim-end"]').value = 0;
      setStatus("Trimmed ✓");
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  editPanel.querySelector('[data-role="save-details"]').addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const title = editPanel.querySelector('[data-role="title-input"]').value.trim();
    const tags = editPanel
      .querySelector('[data-role="tags-input"]')
      .value.split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    btn.disabled = true;
    setStatus("Saving…");
    try {
      const meta = await postJson(`/api/jobs/${jobId}/shorts/${short.index}/update`, { title, tags });
      Object.assign(short, meta);
      refreshResultCard(card, jobId, short);
      setStatus("Saved ✓");
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  editPanel.querySelector('[data-role="emphasis-apply"]').addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const words = editPanel
      .querySelector('[data-role="emphasis-input"]')
      .value.split(",")
      .map((w) => w.trim())
      .filter(Boolean);
    btn.disabled = true;
    setStatus("Re-burning captions…");
    try {
      const meta = await postJson(`/api/jobs/${jobId}/shorts/${short.index}/emphasis`, { words });
      Object.assign(short, meta);
      refreshResultCard(card, jobId, short, { videoChanged: true });
      setStatus("Captions updated ✓");
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  editPanel.querySelector('[data-role="suggest-btn"]').addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const list = editPanel.querySelector('[data-role="suggestions"]');
    btn.disabled = true;
    setStatus("Thinking…");
    list.innerHTML = "";
    try {
      const res = await fetch(`/api/jobs/${jobId}/shorts/${short.index}/suggest-captions`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error((body && body.detail) || `Request failed (${res.status})`);
      }
      const { suggestions } = await res.json();
      if (!suggestions || !suggestions.length) {
        setStatus("No suggestions came back.", true);
        return;
      }
      list.innerHTML = suggestions
        .map(
          (s, i) => `
        <div class="suggestion-card">
          <div class="suggestion-caption">${escapeHtml(s.caption)}</div>
          <div class="suggestion-hashtags">${(s.hashtags || []).map((h) => escapeHtml(h)).join(" ")}</div>
          <button class="suggestion-use-btn" type="button" data-idx="${i}">Use this</button>
        </div>`
        )
        .join("");
      list.querySelectorAll(".suggestion-use-btn").forEach((useBtn) => {
        useBtn.addEventListener("click", async () => {
          const s = suggestions[Number(useBtn.dataset.idx)];
          useBtn.disabled = true;
          setStatus("Applying…");
          try {
            const meta = await postJson(`/api/jobs/${jobId}/shorts/${short.index}/update`, {
              title: s.caption,
              tags: (s.hashtags || []).map((h) => h.replace(/^#/, "")),
            });
            Object.assign(short, meta);
            editPanel.querySelector('[data-role="title-input"]').value = short.title || "";
            editPanel.querySelector('[data-role="tags-input"]').value = (short.tags || []).join(", ");
            refreshResultCard(card, jobId, short);
            setStatus("Applied ✓");
          } catch (err) {
            setStatus(err.message, true);
          } finally {
            useBtn.disabled = false;
          }
        });
      });
      setStatus("");
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });
}

// ---- shared helpers ---------------------------------------------------------

function showOnly(panel) {
  [uploadPanel, progressPanel, reviewPanel, resultsPanel].forEach((p) => {
    p.hidden = p !== panel;
  });
}

function showError(message) {
  showOnly(resultsPanel);
  resultsGrid.innerHTML = `<div class="error-banner">${escapeHtml(message)}</div>`;
}

function resetToUpload() {
  if (currentPollTimer) clearInterval(currentPollTimer);
  currentJobId = null;
  candidates = [];
  selectedFile = null;
  fileInput.value = "";
  urlInput.value = "";
  selectedFileBox.hidden = true;
  dropzone.hidden = false;
  updateStartEnabled();
  setStep(1);
  showOnly(uploadPanel);
}

function formatDuration(sec) {
  if (!sec && sec !== 0) return "";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatElapsed(sec) {
  if (!sec && sec !== 0) return "0s";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const TIMING_LABELS = {
  download_seconds: "download",
  transcribing_seconds: "transcription",
  finding_highlights_seconds: "highlight picking",
};

function formatTimingBreakdown(timing) {
  if (!timing || !Object.keys(timing).length) return "";
  const parts = Object.entries(timing)
    .filter(([, v]) => v > 0.05)
    .map(([k, v]) => `${TIMING_LABELS[k] || k.replace(/_seconds$/, "")}: ${formatElapsed(v)}`);
  return parts.length ? `Timing — ${parts.join(", ")}.` : "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : str;
  return div.innerHTML;
}

function escapeAttr(str) {
  return escapeHtml(str).replaceAll('"', "&quot;");
}
