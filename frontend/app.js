/* global app.js — NBA 2K26 Tendency Generator frontend */

// ── Category display names ────────────────────────────────────────────────
const CATEGORY_LABELS = {
  finishing: "🏀 Finishing",
  shooting: "🎯 Shooting",
  driving: "⚡ Driving",
  dribble_moves: "🔄 Dribble Moves",
  dribble_setup: "🔧 Dribble Setup",
  post: "🏋️ Post Play",
  defense: "🛡️ Defense",
  passing: "🎁 Passing",
  playstyle: "🎮 Play Style",
  triple_threat: "⚡ Triple Threat",
  isolation: "🔥 Isolation",
  physical: "💪 Physical",
  core: "⭐ Core",
  sub_zone: "📍 Sub-Zones",
};

// Preferred category order for display
const CATEGORY_ORDER = [
  "finishing", "shooting", "driving", "dribble_moves", "dribble_setup",
  "post", "defense", "passing", "playstyle", "triple_threat", "isolation",
  "physical", "core", "sub_zone",
];

// ── State ─────────────────────────────────────────────────────────────────
let _currentPlayerData = null;  // last generated single-player API response
let _debounceTimer = null;

// ── DOM refs ──────────────────────────────────────────────────────────────
const playerSearch    = document.getElementById("playerSearch");
const suggestions     = document.getElementById("suggestions");
const seasonSelect    = document.getElementById("seasonSelect");
const teamSelect      = document.getElementById("teamSelect");
const generateTeamBtn = document.getElementById("generateTeamBtn");
const playerResult    = document.getElementById("playerResult");
const playerNameEl    = document.getElementById("playerName");
const playerMetaEl    = document.getElementById("playerMeta");
const tendenciesContainer = document.getElementById("tendenciesContainer");
const spinner         = document.getElementById("spinner");
const spinnerText     = document.getElementById("spinnerText");
const errorBanner     = document.getElementById("errorBanner");
const teamResult      = document.getElementById("teamResult");
const teamTitle       = document.getElementById("teamTitle");
const teamAccordion   = document.getElementById("teamAccordion");
const copyJsonBtn     = document.getElementById("copyJsonBtn");
const dlJsonBtn       = document.getElementById("dlJsonBtn");
const dlCsvBtn        = document.getElementById("dlCsvBtn");
const dlExcelBtn      = document.getElementById("dlExcelBtn");
const showLogBtn      = document.getElementById("showLogBtn");
const logPanel        = document.getElementById("logPanel");
const logContent      = document.getElementById("logContent");
const copyLogBtn      = document.getElementById("copyLogBtn");

// ── Utilities ─────────────────────────────────────────────────────────────
function showSpinner(text = "Loading…") {
  spinnerText.textContent = text;
  spinner.hidden = false;
  playerResult.hidden = true;
  teamResult.hidden = true;
  errorBanner.hidden = true;
}

function hideSpinner() {
  spinner.hidden = true;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

function safeName(name) {
  return name.toLowerCase().replace(/\s+/g, "_");
}

// ── Tendency bar rendering ─────────────────────────────────────────────────
function barClass(value) {
  if (value <= 20) return "low";
  if (value <= 50) return "med";
  return "high";
}

function renderTendencies(tendenciesObj) {
  // Group by category
  const byCategory = {};
  for (const [key, entry] of Object.entries(tendenciesObj)) {
    const cat = entry.category || "core";
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push({ key, label: entry.label, value: entry.value });
  }

  const orderedCats = [
    ...CATEGORY_ORDER.filter(c => byCategory[c]),
    ...Object.keys(byCategory).filter(c => !CATEGORY_ORDER.includes(c)),
  ];

  const html = orderedCats.map(cat => {
    const rows = byCategory[cat].map(t => `
      <div class="tendency-row">
        <span class="tendency-label" title="${t.label}">${t.label}</span>
        <div class="bar-track">
          <div class="bar-fill ${barClass(t.value)}" style="width:${t.value}%"></div>
        </div>
        <span class="tendency-value">${t.value}</span>
      </div>
    `).join("");
    return `
      <div class="category-section">
        <div class="category-title">${CATEGORY_LABELS[cat] || cat}</div>
        ${rows}
      </div>`;
  }).join("");

  tendenciesContainer.innerHTML = html;
}

// ── Player generation ──────────────────────────────────────────────────────
async function generatePlayer(playerName) {
  const season = seasonSelect.value;
  showSpinner(`Generating tendencies for ${playerName}…`);
  try {
    const resp = await fetch(`/generate/${encodeURIComponent(playerName)}?season=${season}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();

    // Merge category into each tendency entry by fetching from API tendencies keys
    // The API already has label per tendency; we need category — fetch registry lazily
    _currentPlayerData = data;

    // Reset log panel for new generation
    if (logPanel) logPanel.hidden = true;
    if (showLogBtn) showLogBtn.textContent = "📊 Show Log";

    playerNameEl.textContent = data.player_name;
    playerMetaEl.textContent = [data.position, data.team, data.season]
      .filter(Boolean).join(" · ");

    // Build enriched tendencies map keyed by label for display
    // The response tendencies are keyed by primjer_key; values have {value, label}
    // We need to attach category info — derive from label groups heuristic or
    // simply display without category (flat list by registry order)
    const enriched = {};
    for (const [key, entry] of Object.entries(data.tendencies)) {
      enriched[key] = { label: entry.label, value: entry.value, category: guessCategoryFromKey(key) };
    }

    renderTendencies(enriched);
    hideSpinner();
    teamResult.hidden = true;
    playerResult.hidden = false;
    hideError();
  } catch (e) {
    hideSpinner();
    showError(`Error: ${e.message}`);
  }
}

// Heuristic: derive category from tendency label / key patterns
function guessCategoryFromKey(key) {
  const k = key.toLowerCase();
  if (k.includes("shot close") || k.includes("shot under") || k.includes("driving layup") ||
      k.includes("step through") || k.includes("euro") || k.includes("alley") ||
      k.includes("putback") || k.includes("up and under") || k.includes("hop step") ||
      k.includes("spin layup") || k.includes("floater") || k.includes("in traffic") ||
      k.includes("finishing")) return "finishing";
  if (k.includes("shot mid") || k.includes("shot three") || k.includes("spot up") ||
      k.includes("pull up") || k.includes("shot off screen") || k.includes("off dribble") ||
      k.includes("catch") || k.includes("standing three") || k.includes("deep three") ||
      k.includes("logo three")) return "shooting";
  if (k.includes("driving") && !k.includes("layup")) return "driving";
  if (k.includes("dribble") && (k.includes("between") || k.includes("behind") ||
      k.includes("crossover") || k.includes("hesitation") || k.includes("in place") ||
      k.includes("combo"))) return "dribble_moves";
  if (k.includes("triple threat") || k.includes("triple-threat")) return "triple_threat";
  if (k.includes("dribble setup") || k.includes("moving shot")) return "dribble_setup";
  if (k.includes("post")) return "post";
  if (k.includes("defense") || k.includes("on ball") || k.includes("off ball") ||
      k.includes("block") || k.includes("steal") || k.includes("help")) return "defense";
  if (k.includes("pass") || k.includes("alley oop") || k.includes("dish")) return "passing";
  if (k.includes("isolation") || k.includes("iso")) return "isolation";
  if (k.includes("speed") || k.includes("strength") || k.includes("acceleration") ||
      k.includes("agility")) return "physical";
  if (k.includes("play style") || k.includes("playstyle") || k.includes("playmaking") ||
      k.includes("ball handler") || k.includes("scorer") || k.includes("slasher")) return "playstyle";
  // sub-zone
  if (k.includes("left") || k.includes("right") || k.includes("middle") ||
      k.includes("center") || k.includes("corner") || k.includes("wing") ||
      k.includes("baseline") || k.includes("elbow") || k.includes("top") ||
      k.includes("short")) return "sub_zone";
  return "core";
}

// ── Export helpers ─────────────────────────────────────────────────────────
function buildJsonPayload(data) {
  // Build primjer.txt-compatible JSON from current data
  const tendencies = {};
  for (const [key, entry] of Object.entries(data.tendencies)) {
    tendencies[key] = {
      value: entry.value,
      label: entry.label,
      offset: entry.offset,
      type: entry.type,
      bit_offset: entry.bit_offset,
      bit_length: entry.bit_length,
      length: entry.length,
    };
  }
  return JSON.stringify({ tendencies }, null, 2);
}

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

copyJsonBtn.addEventListener("click", async () => {
  if (!_currentPlayerData) return;
  try {
    await navigator.clipboard.writeText(buildJsonPayload(_currentPlayerData));
    copyJsonBtn.textContent = "✅ Copied!";
    setTimeout(() => { copyJsonBtn.textContent = "📋 Copy JSON"; }, 2000);
  } catch {
    copyJsonBtn.textContent = "❌ Failed";
    setTimeout(() => { copyJsonBtn.textContent = "📋 Copy JSON"; }, 2000);
  }
});

dlJsonBtn.addEventListener("click", () => {
  if (!_currentPlayerData) return;
  const name = safeName(_currentPlayerData.player_name);
  downloadBlob(buildJsonPayload(_currentPlayerData), `${name}_tendencies.json`, "application/json");
});

dlCsvBtn.addEventListener("click", () => {
  if (!_currentPlayerData) return;
  const season = seasonSelect.value;
  window.location.href = `/export/csv/${encodeURIComponent(_currentPlayerData.player_name)}?season=${season}`;
});

dlExcelBtn.addEventListener("click", () => {
  if (!_currentPlayerData) return;
  const season = seasonSelect.value;
  window.location.href = `/export/excel/${encodeURIComponent(_currentPlayerData.player_name)}?season=${season}`;
});

// ── Log panel ──────────────────────────────────────────────────────────────
const LOG_LABEL_WIDTH = 26;

function buildLogText(data) {
  const status = data.tracking_data_status || {};
  const name = data.player_name || "Unknown";
  const season = data.season || "";
  const position = data.position || "";
  const team = data.team || "";

  const sources = [
    { label: "Play Types (Synergy)", key: "play_types_available" },
    { label: "Tracking Shots",       key: "tracking_shots_available" },
    { label: "Hustle Stats",         key: "hustle_available" },
    { label: "Passing Tracking",     key: "passing_available" },
  ];

  const lines = [];
  lines.push("═══════════════════════════════════════════");
  lines.push(`  DATA SOURCE LOG — ${name}`);
  lines.push(`  Season: ${season} | Position: ${position} | Team: ${team}`);
  lines.push("═══════════════════════════════════════════");
  lines.push("");

  let available = 0;
  for (const s of sources) {
    const ok = status[s.key] === true;
    if (ok) available++;
    const icon = ok ? "✅" : "❌";
    const statusText = ok ? "Available" : "Unavailable";
    const paddedLabel = s.label.padEnd(LOG_LABEL_WIDTH);
    lines.push(`  ${icon} ${paddedLabel} — ${statusText}`);
  }

  lines.push("");
  lines.push("───────────────────────────────────────────");
  lines.push(`  ${available} of ${sources.length} data sources available`);
  if (available < sources.length) {
    lines.push("  Note: Unavailable sources use proxy formulas");
  }
  lines.push("═══════════════════════════════════════════");

  return lines.join("\n");
}

showLogBtn.addEventListener("click", () => {
  if (logPanel.hidden) {
    if (_currentPlayerData) {
      logContent.textContent = buildLogText(_currentPlayerData);
    }
    logPanel.hidden = false;
    showLogBtn.textContent = "📊 Hide Log";
  } else {
    logPanel.hidden = true;
    showLogBtn.textContent = "📊 Show Log";
  }
});

copyLogBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(logContent.textContent);
    copyLogBtn.textContent = "✅ Copied!";
    setTimeout(() => { copyLogBtn.textContent = "📋 Copy Log"; }, 2000);
  } catch {
    copyLogBtn.textContent = "❌ Failed";
    setTimeout(() => { copyLogBtn.textContent = "📋 Copy Log"; }, 2000);
  }
});

// ── Search / Autocomplete ──────────────────────────────────────────────────
playerSearch.addEventListener("input", () => {
  clearTimeout(_debounceTimer);
  const q = playerSearch.value.trim();
  if (q.length < 2) { hideSuggestions(); return; }
  _debounceTimer = setTimeout(() => fetchSuggestions(q), 300);
});

playerSearch.addEventListener("keydown", e => {
  if (e.key === "Enter") {
    const q = playerSearch.value.trim();
    if (q) { hideSuggestions(); generatePlayer(q); }
  }
});

document.addEventListener("click", e => {
  if (!playerSearch.contains(e.target) && !suggestions.contains(e.target)) {
    hideSuggestions();
  }
});

async function fetchSuggestions(query) {
  try {
    const resp = await fetch(`/search/${encodeURIComponent(query)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    showSuggestions(data.results || []);
  } catch { /* ignore search errors */ }
}

function showSuggestions(results) {
  if (!results.length) { hideSuggestions(); return; }
  suggestions.innerHTML = results.slice(0, 8).map(r => `
    <li data-name="${r.full_name}">
      ${r.full_name}
      <span class="sug-team">${r.team || ""}</span>
    </li>
  `).join("");
  suggestions.hidden = false;
  suggestions.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => {
      playerSearch.value = li.dataset.name;
      hideSuggestions();
      generatePlayer(li.dataset.name);
    });
  });
}

function hideSuggestions() {
  suggestions.hidden = true;
  suggestions.innerHTML = "";
}

// ── Team generation ────────────────────────────────────────────────────────
generateTeamBtn.addEventListener("click", async () => {
  const abbr = teamSelect.value;
  if (!abbr) { showError("Please select a team."); return; }
  const season = seasonSelect.value;
  showSpinner(`Generating ${abbr} roster…`);

  try {
    const resp = await fetch(`/team/${abbr}?season=${season}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderTeam(data, season);
    hideSpinner();
    playerResult.hidden = true;
    teamResult.hidden = false;
    hideError();
  } catch (e) {
    hideSpinner();
    showError(`Error: ${e.message}`);
  }
});

function renderTeam(data, season) {
  teamTitle.textContent = `${data.team} — ${data.player_count} players · ${season}`;

  const abbr = encodeURIComponent(data.team_abbr || data.team);
  const teamExportDiv = document.getElementById("teamExportButtons");
  if (teamExportDiv) {
    teamExportDiv.innerHTML = `
      <button class="btn btn-sm" onclick="window.location.href='/export/csv/team/${abbr}?season=${season}'">📥 Export Team CSV</button>
      <button class="btn btn-sm" onclick="window.location.href='/export/excel/team/${abbr}?season=${season}'">📥 Export Team Excel</button>
    `;
  }

  teamAccordion.innerHTML = data.players.map((player, idx) => {
    const enriched = {};
    for (const [key, entry] of Object.entries(player.tendencies)) {
      enriched[key] = { label: entry.label, value: entry.value, category: guessCategoryFromKey(key) };
    }
    const tendencyRows = buildTendencyRowsHtml(enriched);
    const encodedName = encodeURIComponent(player.player_name);
    return `
      <div class="accordion-item" id="acc-${idx}">
        <div class="accordion-header" onclick="toggleAccordion(${idx})">
          <div>
            <span class="accordion-player-name">${player.player_name}</span>
            <span class="accordion-meta">${player.position || ""}</span>
          </div>
          <i class="accordion-chevron">▼</i>
        </div>
        <div class="accordion-body">
          <div class="accordion-export">
            <button class="btn btn-sm" onclick="copyTeamPlayerJson(${idx})">📋 Copy JSON</button>
            <button class="btn btn-sm" onclick="window.location.href='/export/csv/${encodedName}?season=${season}'">📥 CSV</button>
            <button class="btn btn-sm" onclick="window.location.href='/export/excel/${encodedName}?season=${season}'">📥 Excel</button>
          </div>
          <div id="acc-body-${idx}">${tendencyRows}</div>
        </div>
      </div>`;
  }).join("");

  // Store team data for JSON copy
  window._teamData = data;
}

function buildTendencyRowsHtml(enriched) {
  const byCategory = {};
  for (const [key, entry] of Object.entries(enriched)) {
    const cat = entry.category || "core";
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push({ key, label: entry.label, value: entry.value });
  }
  const orderedCats = [
    ...CATEGORY_ORDER.filter(c => byCategory[c]),
    ...Object.keys(byCategory).filter(c => !CATEGORY_ORDER.includes(c)),
  ];
  return orderedCats.map(cat => {
    const rows = byCategory[cat].map(t => `
      <div class="tendency-row">
        <span class="tendency-label" title="${t.label}">${t.label}</span>
        <div class="bar-track">
          <div class="bar-fill ${barClass(t.value)}" style="width:${t.value}%"></div>
        </div>
        <span class="tendency-value">${t.value}</span>
      </div>`).join("");
    return `<div class="category-section">
      <div class="category-title">${CATEGORY_LABELS[cat] || cat}</div>
      ${rows}
    </div>`;
  }).join("");
}

function toggleAccordion(idx) {
  const item = document.getElementById(`acc-${idx}`);
  item.classList.toggle("open");
}

function copyTeamPlayerJson(idx) {
  if (!window._teamData) return;
  const player = window._teamData.players[idx];
  if (!player) return;
  const tendencies = {};
  for (const [key, entry] of Object.entries(player.tendencies)) {
    tendencies[key] = {
      value: entry.value,
      label: entry.label,
      offset: entry.offset,
      type: entry.type,
      bit_offset: entry.bit_offset,
      bit_length: entry.bit_length,
      length: entry.length,
    };
  }
  const json = JSON.stringify({ tendencies }, null, 2);
  navigator.clipboard.writeText(json).catch(() => {});
}
