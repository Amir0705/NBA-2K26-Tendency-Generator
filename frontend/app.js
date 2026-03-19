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

// Attribute category display names & order
const ATTR_CATEGORY_LABELS = {
  finishing:  "🏀 Finishing",
  shooting:   "🎯 Shooting",
  post_game:  "🏋️ Post Game",
  playmaking: "🎁 Playmaking",
  mental:     "🧠 Mental",
  defense:    "🛡️ Defense",
  rebounding: "📊 Rebounding",
  physical:   "💪 Physical",
  meta:       "⭐ Meta",
};
const ATTR_CATEGORY_ORDER = [
  "finishing", "shooting", "post_game", "playmaking", "mental",
  "defense", "rebounding", "physical", "meta",
];

// ── State ─────────────────────────────────────────────────────────────────
let _currentPlayerData = null;  // last generated single-player API response
let _debounceTimer = null;
let _suggestionIndex = -1;
let _latestSuggestions = [];
const _recentKey = "recentPlayers";
let _compareDebounceTimer = null;
let _compareSuggestionIndex = -1;
let _compareSuggestions = [];
let _authToken = localStorage.getItem("authToken") || "";
let _authUser = null;
let _tendencyOriginalValues = {};

const _ROLE_VIEWER = "viewer";
const _ROLE_EDITOR = "editor";
const _ROLE_ADMIN = "admin";
const _SEASON_MIN = 2000;
const _SEASON_MAX = 2025;

// ── DOM refs ──────────────────────────────────────────────────────────────
const playerSearch    = document.getElementById("playerSearch");
const generatePlayerBtn = document.getElementById("generatePlayerBtn");
const suggestions     = document.getElementById("suggestions");
const seasonSelect     = document.getElementById("seasonSelect");
const teamSeasonSelect = document.getElementById("teamSeasonSelect");
const teamSelect       = document.getElementById("teamSelect");
const generateTeamBtn  = document.getElementById("generateTeamBtn");
const recentPlayersEl = document.getElementById("recentPlayers");
const playerResult    = document.getElementById("playerResult");
const playerNameEl    = document.getElementById("playerName");
const playerMetaEl    = document.getElementById("playerMeta");
const dataModeBadge   = document.getElementById("dataModeBadge");
const playStylesSection = document.getElementById("playStylesSection");
const playStylesContainer = document.getElementById("playStylesContainer");
const guardrailBadge  = document.getElementById("guardrailBadge");
const errorBadge      = document.getElementById("errorBadge");
const tendenciesContainer = document.getElementById("tendenciesContainer");
const attributesContainer = document.getElementById("attributesContainer");
const viewToggle      = document.getElementById("viewToggle");
const spinner         = document.getElementById("spinner");
const spinnerText     = document.getElementById("spinnerText");
const errorBanner     = document.getElementById("errorBanner");
const teamResult      = document.getElementById("teamResult");
const teamTitle       = document.getElementById("teamTitle");
const teamAccordion   = document.getElementById("teamAccordion");
const teamRosterGrid  = document.getElementById("teamRosterGrid");
const teamExpandedPlayerDetail = document.getElementById("teamExpandedPlayerDetail");
const teamExpandedContent = document.getElementById("teamExpandedContent");
const teamCloseExpandedBtn = document.getElementById("teamCloseExpandedBtn");
const dl2kJsonBtn     = document.getElementById("dl2kJsonBtn");
const dlExcelBtn      = document.getElementById("dlExcelBtn");
const toggleDebugBtn  = document.getElementById("toggleDebugBtn");
const debugPanel      = document.getElementById("debugPanel");
const comparePlayerSearch = document.getElementById("comparePlayerSearch");
const compareSuggestions = document.getElementById("compareSuggestions");
const compareBtn      = document.getElementById("compareBtn");
const compareTitle    = document.getElementById("compareTitle");
const compareTable    = document.getElementById("compareTable");
const authLoggedIn    = document.getElementById("authLoggedIn");
const logoutBtn       = document.getElementById("logoutBtn");
const authWelcome     = document.getElementById("authWelcome");
const authRole        = document.getElementById("authRole");
const adminUsersBtn   = document.getElementById("adminUsersBtn");
const editorFeedbackSection = document.getElementById("editorFeedbackSection");
const editorFeedbackNotes = document.getElementById("editorFeedbackNotes");
const saveFeedbackBtn = document.getElementById("saveFeedbackBtn");
const editorFeedbackStatus = document.getElementById("editorFeedbackStatus");
const profileCard = document.getElementById("profileCard");
const profileCardName = document.getElementById("profileCardName");
const profileCardMeta = document.getElementById("profileCardMeta");
const profileCardOVR = document.getElementById("profileCardOVR");
const profileCardStats = document.getElementById("profileCardStats");
const profileOffenseBadge = document.getElementById("profileOffenseBadge");
const profileDefenseBadge = document.getElementById("profileDefenseBadge");
const profilePhysicalsBadge = document.getElementById("profilePhysicalsBadge");
const profileViewTendenciesBtn = document.getElementById("profileViewTendenciesBtn");
const profileViewAttributesBtn = document.getElementById("profileViewAttributesBtn");
const profileTendenciesPanel = document.getElementById("profileTendenciesPanel");
const profileAttributesPanel = document.getElementById("profileAttributesPanel");
const profileTendenciesContent = document.getElementById("profileTendenciesContent");
const profileAttributesContent = document.getElementById("profileAttributesContent");

function canEditTendencies() {
  if (!_authUser) return false;
  return _authUser.role === _ROLE_EDITOR || _authUser.role === _ROLE_ADMIN;
}

function isAdmin() {
  return !!_authUser && _authUser.role === _ROLE_ADMIN;
}

// ── Utilities ─────────────────────────────────────────────────────────────
async function apiFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (_authToken) {
    headers.Authorization = `Bearer ${_authToken}`;
  }

  const response = await fetch(url, { ...options, headers });
  if (response.status === 401) {
    _authToken = "";
    _authUser = null;
    localStorage.removeItem("authToken");
    applyAuthState();
    showError("Your session expired. Please log in again.");
  }
  return response;
}

function showSpinner(text = "Loading…") {
  spinnerText.textContent = text;
  spinner.hidden = false;
  playerResult.hidden = true;
  teamResult.hidden = true;
  errorBanner.hidden = true;
  setBusyState(true);
}

function hideSpinner() {
  spinner.hidden = true;
  setBusyState(false);
}

function setBusyState(isBusy) {
  playerSearch.disabled = isBusy;
  comparePlayerSearch.disabled = isBusy;
  generatePlayerBtn.disabled = isBusy;
  compareBtn.disabled = isBusy;
  seasonSelect.disabled = isBusy;
  teamSeasonSelect.disabled = isBusy;
  teamSelect.disabled = isBusy;
  generateTeamBtn.disabled = isBusy;
}

function applyAuthState() {
  const loggedIn = !!_authUser;
  const mustChange = loggedIn && !!_authUser.must_change_password;

  if (authLoggedIn) authLoggedIn.hidden = !loggedIn;

  if (_authUser) {
    if (authWelcome) authWelcome.textContent = `Welcome, ${_authUser.full_name || _authUser.username}`;
    if (authRole) authRole.textContent = `Role: ${_authUser.role}`;
  } else {
    if (authWelcome) authWelcome.textContent = "";
    if (authRole) authRole.textContent = "";
  }

  if (adminUsersBtn) adminUsersBtn.hidden = !isAdmin();
  if (editorFeedbackSection) editorFeedbackSection.hidden = !canEditTendencies();

  const disableApp = !loggedIn || mustChange;
  playerSearch.disabled = disableApp;
  comparePlayerSearch.disabled = disableApp;
  generatePlayerBtn.disabled = disableApp;
  compareBtn.disabled = disableApp;
  seasonSelect.disabled = disableApp;
  teamSeasonSelect.disabled = disableApp;
  teamSelect.disabled = disableApp;
  generateTeamBtn.disabled = disableApp;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

function buildSeasonLabel(startYear) {
  const end = String(startYear + 1).slice(-2);
  return `${startYear}-${end}`;
}

function populateSeasonOptions() {
  // Player generation starts at _SEASON_MAX - 1 (2024-25) because the current
  // season is unfinished; use the team selector if you need 2025-26 rosters.
  const _PLAYER_SEASON_MAX = _SEASON_MAX - 1;
  const current = seasonSelect.value || buildSeasonLabel(_PLAYER_SEASON_MAX);
  seasonSelect.innerHTML = "";
  for (let year = _PLAYER_SEASON_MAX; year >= _SEASON_MIN; year -= 1) {
    const season = buildSeasonLabel(year);
    const opt = document.createElement("option");
    opt.value = season;
    opt.textContent = season;
    seasonSelect.appendChild(opt);
  }
  if (Array.from(seasonSelect.options).some(o => o.value === current)) {
    seasonSelect.value = current;
  }

  // Populate team season selector with the same range.
  const teamCurrent = teamSeasonSelect.value || buildSeasonLabel(_SEASON_MAX);
  teamSeasonSelect.innerHTML = "";
  for (let year = _SEASON_MAX; year >= _SEASON_MIN; year -= 1) {
    const season = buildSeasonLabel(year);
    const opt = document.createElement("option");
    opt.value = season;
    // For 2025-26 clarify that stats come from 2024-25
    opt.textContent = year === _SEASON_MAX ? `${season} (stats from ${buildSeasonLabel(year - 1)})` : season;
    teamSeasonSelect.appendChild(opt);
  }
  if (Array.from(teamSeasonSelect.options).some(o => o.value === teamCurrent)) {
    teamSeasonSelect.value = teamCurrent;
  }
}

function dataModeForSeason(season) {
  const start = Number(String(season || "").slice(0, 4));
  if (!Number.isFinite(start)) return "Unknown";
  return start >= 2016 ? "PBP Enhanced" : "Legacy Mode";
}

function updateDataModeBadge(season) {
  if (!dataModeBadge) return;
  dataModeBadge.textContent = dataModeForSeason(season);
  dataModeBadge.hidden = false;
}

function safeName(name) {
  return name.toLowerCase().replace(/\s+/g, "_");
}

function setMetaBadges(data) {
  const debug = data.debug || {};
  const guardrails = Number(debug.guardrail_count || 0);
  const errors = Number(debug.error_count || 0);

  if (guardrails > 0) {
    guardrailBadge.textContent = `Guardrails: ${guardrails}`;
    guardrailBadge.hidden = false;
  } else {
    guardrailBadge.hidden = true;
  }

  if (errors > 0) {
    errorBadge.textContent = `Errors: ${errors}`;
    errorBadge.hidden = false;
  } else {
    errorBadge.hidden = true;
  }

  debugPanel.textContent = JSON.stringify(debug, null, 2);
}

// ── Tendency bar rendering ─────────────────────────────────────────────────
function barClass(value) {
  if (value <= 20) return "low";
  if (value <= 50) return "med";
  return "high";
}

function renderTendencies(tendenciesObj) {
  _tendencyOriginalValues = {};
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
        ${canEditTendencies() ? `<input type="number" min="0" max="100" class="tendency-edit" data-key="${t.key}" value="${t.value}" />` : ""}
      </div>
    `).join("");
    return `
      <div class="category-section">
        <div class="category-title">${CATEGORY_LABELS[cat] || cat}</div>
        ${rows}
      </div>`;
  }).join("");

  tendenciesContainer.innerHTML = html;
  if (canEditTendencies()) {
    tendenciesContainer.querySelectorAll(".tendency-edit").forEach((inputEl) => {
      const key = inputEl.dataset.key;
      _tendencyOriginalValues[key] = Number(inputEl.value || 0);
      inputEl.addEventListener("input", () => {
        const now = Number(inputEl.value || 0);
        const base = Number(_tendencyOriginalValues[key] || 0);
        inputEl.classList.toggle("changed", now !== base);
      });
    });
  }
}

function buildPlayStylesHtml(priorities, weights, usageRate) {
  const safePriorities = Array.isArray(priorities) ? priorities : [];
  if (!safePriorities.length) {
    return '<div class="playstyles-empty">No play styles were selected.</div>';
  }

  const usageText = Number.isFinite(Number(usageRate))
    ? `<div class="playstyles-meta">Usage rate: ${Number(usageRate).toFixed(1)}%</div>`
    : "";

  const rows = safePriorities.map((style, idx) => {
    const weight = Number((weights || {})[style] || 0);
    const pct = Math.round(weight * 100);
    return `
      <div class="playstyle-row">
        <span class="playstyle-rank">${idx + 1}</span>
        <span class="playstyle-name">${style}</span>
        <span class="playstyle-weight">${pct}%</span>
      </div>
    `;
  }).join("");

  return `${usageText}<div class="playstyles-list">${rows}</div>`;
}

function renderPlayStyles(data) {
  const priorities = data.play_style_priorities || [];
  const weights = data.play_style_weights || {};
  const usageRate = data.play_style_usage_rate;

  playStylesContainer.innerHTML = buildPlayStylesHtml(priorities, weights, usageRate);
  playStylesSection.hidden = false;
}

// ── Attribute bar rendering ────────────────────────────────────────────────
function attrTier(value) {
  if (value >= 85) return "elite";
  if (value >= 70) return "good";
  if (value >= 50) return "average";
  return "poor";
}

function renderAttributes(attrsObj) {
  if (!attrsObj || !Object.keys(attrsObj).length) {
    attributesContainer.innerHTML = '<div class="playstyles-empty">No attribute data available.</div>';
    return;
  }
  const byCategory = {};
  for (const [key, entry] of Object.entries(attrsObj)) {
    const cat = entry.category || "meta";
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push({ key, label: entry.label || key, value: entry.value });
  }

  const orderedCats = [
    ...ATTR_CATEGORY_ORDER.filter(c => byCategory[c]),
    ...Object.keys(byCategory).filter(c => !ATTR_CATEGORY_ORDER.includes(c)),
  ];

  const html = orderedCats.map(cat => {
    const rows = byCategory[cat].map(a => {
      const pct = Math.round(((a.value - 25) / 74) * 100);
      const tier = attrTier(a.value);
      return `
        <div class="tendency-row">
          <span class="tendency-label" title="${a.label}">${a.label}</span>
          <div class="bar-track">
            <div class="bar-fill ${tier === "elite" ? "high" : tier === "good" ? "high" : tier === "average" ? "med" : "low"}" style="width:${pct}%"></div>
          </div>
          <span class="attr-value ${tier}">${a.value}</span>
        </div>`;
    }).join("");
    return `
      <div class="category-section">
        <div class="category-title">${ATTR_CATEGORY_LABELS[cat] || cat}</div>
        ${rows}
      </div>`;
  }).join("");

  attributesContainer.innerHTML = html;
}

// ── View toggle wiring ────────────────────────────────────────────────────
viewToggle.addEventListener("click", e => {
  const btn = e.target.closest(".toggle-btn");
  if (!btn) return;
  const view = btn.dataset.view;
  viewToggle.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  tendenciesContainer.hidden = (view !== "tendencies");
  attributesContainer.hidden = (view !== "attributes");
});

// ── Player generation ──────────────────────────────────────────────────────
async function generatePlayer(playerName) {
  if (!_authUser || _authUser.must_change_password) {
    showError("Please log in first.");
    return;
  }
  if (!playerName || !playerName.trim()) {
    showError("Please enter a player name.");
    return;
  }
  const cleanedPlayerName = playerName.trim();
  const season = seasonSelect.value;
  showSpinner(`Generating tendencies for ${cleanedPlayerName}…`);
  try {
    const resp = await apiFetch(`/generate/${encodeURIComponent(cleanedPlayerName)}?season=${season}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();

    // Merge category into each tendency entry by fetching from API tendencies keys
    // The API already has label per tendency; we need category — fetch registry lazily
    _currentPlayerData = data;

    playerNameEl.textContent = data.player_name;
    playerMetaEl.textContent = [data.position, data.team, data.season]
      .filter(Boolean).join(" · ");
    updateDataModeBadge(data.season || season);
    renderPlayStyles(data);

    // Build enriched tendencies map keyed by label for display
    // The response tendencies are keyed by primjer_key; values have {value, label}
    // We need to attach category info — derive from label groups heuristic or
    // simply display without category (flat list by registry order)
    const enriched = {};
    for (const [key, entry] of Object.entries(data.tendencies)) {
      enriched[key] = { label: entry.label, value: entry.value, category: guessCategoryFromKey(key) };
    }

    // Render Phase 1 profile card
    renderProfileCard(data);
    initializeCollapsiblePanels(data);

    renderTendencies(enriched);
    renderAttributes(data.attributes || {});
    viewToggle.hidden = false;
    // Reset toggle to Tendencies view
    viewToggle.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
    viewToggle.querySelector('[data-view="tendencies"]').classList.add("active");
    tendenciesContainer.hidden = false;
    attributesContainer.hidden = true;
    setMetaBadges(data);
    addRecentPlayer(data.player_name || cleanedPlayerName);
    hideSpinner();
    teamResult.hidden = true;
    playerResult.hidden = false;
    compareTitle.hidden = true;
    compareTable.hidden = true;
    compareTable.innerHTML = "";
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

// ── Profile Card Rendering ─────────────────────────────────────────────────
function calculateSummaryGrades(tendencies, attributes) {
  // Calculate Offense grade based on shooting and driving tendencies
  const offenseKeys = Object.keys(tendencies).filter(k => {
    const cat = guessCategoryFromKey(k);
    return cat === "shooting" || cat === "driving" || cat === "finishing";
  });
  const offenseAvg = offenseKeys.length > 0
    ? offenseKeys.reduce((sum, k) => sum + (tendencies[k]?.value || 0), 0) / offenseKeys.length
    : 50;

  // Calculate Defense grade based on defense tendencies
  const defenseKeys = Object.keys(tendencies).filter(k => guessCategoryFromKey(k) === "defense");
  const defenseAvg = defenseKeys.length > 0
    ? defenseKeys.reduce((sum, k) => sum + (tendencies[k]?.value || 0), 0) / defenseKeys.length
    : 50;

  // Calculate Physicals grade based on physical attributes
  const physicalAttrs = Object.values(attributes).filter(a => a.category === "physical");
  const physicalAvg = physicalAttrs.length > 0
    ? physicalAttrs.reduce((sum, a) => sum + a.value, 0) / physicalAttrs.length
    : 50;

  const getGrade = (value) => {
    if (value >= 85) return "S";
    if (value >= 80) return "A+";
    if (value >= 75) return "A";
    if (value >= 70) return "B+";
    if (value >= 60) return "B";
    return "C+";
  };

  return {
    offense: getGrade(offenseAvg),
    defense: getGrade(defenseAvg),
    physicals: getGrade(physicalAvg),
  };
}

function getProductionStats(playerData) {
  // Extract sample production stats from player data if available
  // These would typically come from the API response or season data
  const stats = {
    ppg: playerData.ppg || "N/A",
    apg: playerData.apg || "N/A",
    rpg: playerData.rpg || "N/A",
  };
  return stats;
}

function renderProfileCard(playerData) {
  // Render profile card header
  profileCardName.textContent = playerData.player_name || "Unknown";
  const metaParts = [playerData.position, playerData.team, playerData.season].filter(Boolean);
  profileCardMeta.textContent = metaParts.join(" · ");

  // Calculate OVR (average of all attributes or from data)
  const allAttrs = Object.values(playerData.attributes || {});
  const ovrValue = allAttrs.length > 0
    ? Math.round(allAttrs.reduce((sum, a) => sum + a.value, 0) / allAttrs.length)
    : 75;
  profileCardOVR.textContent = ovrValue;

  // Render summary badges
  const grades = calculateSummaryGrades(playerData.tendencies || {}, playerData.attributes || {});
  [
    { badge: profileOffenseBadge, label: "Offense", grade: grades.offense },
    { badge: profileDefenseBadge, label: "Defense", grade: grades.defense },
    { badge: profilePhysicalsBadge, label: "Physicals", grade: grades.physicals },
  ].forEach(({ badge, label, grade }) => {
    badge.innerHTML = `
      <div class="summary-badge-label">${label}</div>
      <div class="summary-badge-grade">${grade}</div>
    `;
  });

  // Render stats strip
  const stats = getProductionStats(playerData);
  profileCardStats.innerHTML = `
    <div class="stat-item">
      <span class="stat-label">PPG</span>
      <span class="stat-value">${stats.ppg}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">APG</span>
      <span class="stat-value">${stats.apg}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">RPG</span>
      <span class="stat-value">${stats.rpg}</span>
    </div>
  `;
}

function initializeCollapsiblePanels(playerData) {
  // Set up event listeners for collapsible panels
  const headers = document.querySelectorAll(".collapsible-header");
  headers.forEach(header => {
    header.addEventListener("click", () => {
      const panel = header.closest(".collapsible-panel");
      panel.classList.toggle("open");
    });
  });

  // Populate tendencies content
  const enrichedTendencies = {};
  for (const [key, entry] of Object.entries(playerData.tendencies || {})) {
    enrichedTendencies[key] = {
      label: entry.label,
      value: entry.value,
      category: guessCategoryFromKey(key),
    };
  }
  const tendenciesHtml = renderTendenciesToHtml(enrichedTendencies);
  profileTendenciesContent.innerHTML = tendenciesHtml;

  // Populate attributes content
  const attributesHtml = renderAttributesToHtml(playerData.attributes || {});
  profileAttributesContent.innerHTML = attributesHtml;

  // Button handlers
  profileViewTendenciesBtn.addEventListener("click", () => {
    profileTendenciesPanel.classList.remove("hidden");
    profileTendenciesPanel.classList.add("open");
    profileAttributesPanel.classList.add("hidden");
    profileAttributesPanel.classList.remove("open");
  });

  profileViewAttributesBtn.addEventListener("click", () => {
    profileAttributesPanel.classList.remove("hidden");
    profileAttributesPanel.classList.add("open");
    profileTendenciesPanel.classList.add("hidden");
    profileTendenciesPanel.classList.remove("open");
  });
}

// Helper to render tendencies as HTML for profile card
function renderTendenciesToHtml(tendenciesObj) {
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

  return orderedCats.map(cat => {
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
}

// Helper to render attributes as HTML for profile card
function renderAttributesToHtml(attrsObj) {
  if (!attrsObj || !Object.keys(attrsObj).length) {
    return '<div class="playstyles-empty">No attribute data available.</div>';
  }
  const byCategory = {};
  for (const [key, entry] of Object.entries(attrsObj)) {
    const cat = entry.category || "meta";
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push({ key, label: entry.label || key, value: entry.value });
  }

  const orderedCats = [
    ...ATTR_CATEGORY_ORDER.filter(c => byCategory[c]),
    ...Object.keys(byCategory).filter(c => !ATTR_CATEGORY_ORDER.includes(c)),
  ];

  return orderedCats.map(cat => {
    const rows = byCategory[cat].map(a => {
      const pct = Math.round(((a.value - 25) / 74) * 100);
      const tier = attrTier(a.value);
      return `
        <div class="tendency-row">
          <span class="tendency-label" title="${a.label}">${a.label}</span>
          <div class="bar-track">
            <div class="bar-fill ${tier === "elite" ? "high" : tier === "good" ? "high" : tier === "average" ? "med" : "low"}" style="width:${pct}%"></div>
          </div>
          <span class="attr-value ${tier}">${a.value}</span>
        </div>`;
    }).join("");
    return `
      <div class="category-section">
        <div class="category-title">${ATTR_CATEGORY_LABELS[cat] || cat}</div>
        ${rows}
      </div>`;
  }).join("");
}

// ── Export helpers ─────────────────────────────────────────────────────────
dl2kJsonBtn.addEventListener("click", () => {
  if (!_currentPlayerData) return;
  const season = seasonSelect.value;
  window.location.href = `/export/2k/${encodeURIComponent(_currentPlayerData.player_name)}?season=${season}`;
});

dlExcelBtn.addEventListener("click", () => {
  if (!_currentPlayerData) return;
  const season = seasonSelect.value;
  window.location.href = `/export/excel/${encodeURIComponent(_currentPlayerData.player_name)}?season=${season}`;
});

toggleDebugBtn.addEventListener("click", () => {
  debugPanel.hidden = !debugPanel.hidden;
  toggleDebugBtn.textContent = debugPanel.hidden ? "🛠 Debug" : "🙈 Hide Debug";
});

// ── Search / Autocomplete ──────────────────────────────────────────────────
playerSearch.addEventListener("input", () => {
  clearTimeout(_debounceTimer);
  const q = playerSearch.value.trim();
  if (q.length < 2) { hideSuggestions(); return; }
  _debounceTimer = setTimeout(() => fetchSuggestions(q), 300);
});

playerSearch.addEventListener("keydown", e => {
  if (e.key === "ArrowDown" && _latestSuggestions.length > 0) {
    e.preventDefault();
    _suggestionIndex = Math.min(_suggestionIndex + 1, _latestSuggestions.length - 1);
    updateSuggestionActiveState();
    return;
  }
  if (e.key === "ArrowUp" && _latestSuggestions.length > 0) {
    e.preventDefault();
    _suggestionIndex = Math.max(_suggestionIndex - 1, 0);
    updateSuggestionActiveState();
    return;
  }
  if (e.key === "Enter") {
    e.preventDefault();
    if (_suggestionIndex >= 0 && _latestSuggestions[_suggestionIndex]) {
      const selected = _latestSuggestions[_suggestionIndex].full_name;
      playerSearch.value = selected;
      hideSuggestions();
      generatePlayer(selected);
      return;
    }
    const q = playerSearch.value.trim();
    if (q) { hideSuggestions(); generatePlayer(q); }
  }
  if (e.key === "Escape") {
    hideSuggestions();
  }
});

generatePlayerBtn.addEventListener("click", () => {
  const q = playerSearch.value.trim();
  if (q) {
    hideSuggestions();
    generatePlayer(q);
  } else {
    showError("Please enter a player name.");
  }
});

document.addEventListener("click", e => {
  if (!playerSearch.contains(e.target) && !suggestions.contains(e.target)) {
    hideSuggestions();
  }
  if (!comparePlayerSearch.contains(e.target) && !compareSuggestions.contains(e.target)) {
    hideCompareSuggestions();
  }
});

async function fetchSuggestions(query) {
  try {
    const resp = await apiFetch(`/search/${encodeURIComponent(query)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    showSuggestions(data.results || []);
  } catch { /* ignore search errors */ }
}

function showSuggestions(results) {
  if (!results.length) { hideSuggestions(); return; }
  _latestSuggestions = results.slice(0, 8);
  _suggestionIndex = -1;
  suggestions.innerHTML = _latestSuggestions.map((r, idx) => `
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
  _latestSuggestions = [];
  _suggestionIndex = -1;
  suggestions.hidden = true;
  suggestions.innerHTML = "";
}

function updateSuggestionActiveState() {
  const items = suggestions.querySelectorAll("li");
  items.forEach((li, idx) => {
    li.classList.toggle("active", idx === _suggestionIndex);
  });
}

comparePlayerSearch.addEventListener("input", () => {
  clearTimeout(_compareDebounceTimer);
  const q = comparePlayerSearch.value.trim();
  if (q.length < 2) { hideCompareSuggestions(); return; }
  _compareDebounceTimer = setTimeout(() => fetchCompareSuggestions(q), 250);
});

comparePlayerSearch.addEventListener("keydown", e => {
  if (e.key === "ArrowDown" && _compareSuggestions.length > 0) {
    e.preventDefault();
    _compareSuggestionIndex = Math.min(_compareSuggestionIndex + 1, _compareSuggestions.length - 1);
    updateCompareSuggestionActiveState();
    return;
  }
  if (e.key === "ArrowUp" && _compareSuggestions.length > 0) {
    e.preventDefault();
    _compareSuggestionIndex = Math.max(_compareSuggestionIndex - 1, 0);
    updateCompareSuggestionActiveState();
    return;
  }
  if (e.key === "Enter") {
    e.preventDefault();
    if (_compareSuggestionIndex >= 0 && _compareSuggestions[_compareSuggestionIndex]) {
      comparePlayerSearch.value = _compareSuggestions[_compareSuggestionIndex].full_name;
      hideCompareSuggestions();
    }
    comparePlayers();
  }
  if (e.key === "Escape") {
    hideCompareSuggestions();
  }
});

async function fetchCompareSuggestions(query) {
  try {
    const resp = await apiFetch(`/search/${encodeURIComponent(query)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    showCompareSuggestions(data.results || []);
  } catch {
    hideCompareSuggestions();
  }
}

function showCompareSuggestions(results) {
  if (!results.length) { hideCompareSuggestions(); return; }
  _compareSuggestions = results.slice(0, 8);
  _compareSuggestionIndex = -1;
  compareSuggestions.innerHTML = _compareSuggestions.map(r => `
    <li data-name="${r.full_name}">
      ${r.full_name}
      <span class="sug-team">${r.team || ""}</span>
    </li>
  `).join("");
  compareSuggestions.hidden = false;
  compareSuggestions.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => {
      comparePlayerSearch.value = li.dataset.name;
      hideCompareSuggestions();
      comparePlayers();
    });
  });
}

function hideCompareSuggestions() {
  _compareSuggestions = [];
  _compareSuggestionIndex = -1;
  compareSuggestions.hidden = true;
  compareSuggestions.innerHTML = "";
}

function updateCompareSuggestionActiveState() {
  const items = compareSuggestions.querySelectorAll("li");
  items.forEach((li, idx) => {
    li.classList.toggle("active", idx === _compareSuggestionIndex);
  });
}

function addRecentPlayer(name) {
  if (!name) return;
  const recent = JSON.parse(localStorage.getItem(_recentKey) || "[]");
  const next = [name, ...recent.filter(n => n !== name)].slice(0, 6);
  localStorage.setItem(_recentKey, JSON.stringify(next));
  renderRecentPlayers();
}

function renderRecentPlayers() {
  const recent = JSON.parse(localStorage.getItem(_recentKey) || "[]");
  if (!recent.length) {
    recentPlayersEl.hidden = true;
    recentPlayersEl.innerHTML = "";
    return;
  }
  recentPlayersEl.hidden = false;
  recentPlayersEl.innerHTML = recent
    .map(name => `<button class="recent-chip" data-name="${name}">${name}</button>`)
    .join("");

  recentPlayersEl.querySelectorAll(".recent-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.name;
      playerSearch.value = name;
      hideSuggestions();
      generatePlayer(name);
    });
  });
}

renderRecentPlayers();

async function comparePlayers() {
  if (!_authUser || _authUser.must_change_password) {
    showError("Please log in first.");
    return;
  }
  if (!_currentPlayerData) {
    showError("Generate a base player first.");
    return;
  }
  const name = comparePlayerSearch.value.trim();
  if (!name) {
    showError("Enter a player name to compare.");
    return;
  }

  showSpinner(`Comparing with ${name}…`);
  try {
    const season = seasonSelect.value;
    const resp = await apiFetch(`/generate/${encodeURIComponent(name)}?season=${season}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const compareData = await resp.json();

    const left = _currentPlayerData.tendencies || {};
    const right = compareData.tendencies || {};
    const rows = Object.keys(left)
      .filter(k => right[k])
      .map(k => {
        const a = Number(left[k].value || 0);
        const b = Number(right[k].value || 0);
        return {
          label: left[k].label || k,
          leftVal: a,
          rightVal: b,
          delta: b - a,
        };
      })
      .sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta))
      .slice(0, 24);

    compareTitle.textContent = `${_currentPlayerData.player_name} vs ${compareData.player_name} (Top Δ tendencies)`;
    compareTable.innerHTML = [
      `<div class="compare-row header"><span>Tendency</span><span>${_currentPlayerData.player_name}</span><span>${compareData.player_name}</span><span>Δ</span></div>`,
      ...rows.map(r => `<div class="compare-row"><span>${r.label}</span><span>${r.leftVal}</span><span>${r.rightVal}</span><span class="compare-delta ${r.delta >= 0 ? "pos" : "neg"}">${r.delta >= 0 ? "+" : ""}${r.delta}</span></div>`),
    ].join("");

    hideSpinner();
    playerResult.hidden = false;
    compareTitle.hidden = false;
    compareTable.hidden = false;
    teamResult.hidden = true;
    hideError();
  } catch (e) {
    hideSpinner();
    showError(`Error: ${e.message}`);
  }
}

compareBtn.addEventListener("click", comparePlayers);

// ── Team generation ────────────────────────────────────────────────────────
generateTeamBtn.addEventListener("click", async () => {
  if (!_authUser || _authUser.must_change_password) {
    showError("Please log in first.");
    return;
  }
  const abbr = teamSelect.value;
  if (!abbr) { showError("Please select a team."); return; }
  const selectedSeason = teamSeasonSelect.value;

  // 2025-26 is the current, unfinished season — use 2024-25 stats with
  // the 2025-26 roster so rosters are current but stats are reliable.
  const IS_CURRENT_UNFINISHED = selectedSeason === buildSeasonLabel(_SEASON_MAX);
  const statsSeason  = IS_CURRENT_UNFINISHED ? buildSeasonLabel(_SEASON_MAX - 1) : selectedSeason;
  const rosterSeason = selectedSeason;  // always use the selected roster

  showSpinner(`Generating ${abbr} roster…`);

  try {
    const resp = await apiFetch(`/team/${abbr}?season=${statsSeason}&roster_season=${rosterSeason}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderTeam(data, selectedSeason, statsSeason);
    hideSpinner();
    playerResult.hidden = true;
    teamResult.hidden = false;
    hideError();
  } catch (e) {
    hideSpinner();
    showError(`Error: ${e.message}`);
  }
});

function renderTeam(data, season, statsSeason) {
  const generated = Number(data.generated_count ?? data.player_count ?? 0);
  const total = Number(data.total_players ?? data.player_count ?? generated);
  const rosterSeason = data.roster_season || season;
  const _statsSeason = statsSeason || data.season || season;
  const mode = data.data_mode ? (data.data_mode === "pbp" ? "PBP Enhanced" : "Legacy Mode") : dataModeForSeason(_statsSeason);
  const statsLabel = _statsSeason !== rosterSeason
    ? `stats ${_statsSeason} · roster ${rosterSeason}`
    : `season ${rosterSeason}`;
  teamTitle.textContent = `${data.team} — generated ${generated}/${total} · ${statsLabel} · ${mode}`;

  const abbr = encodeURIComponent(data.team_abbr || data.team);
  const teamExportDiv = document.getElementById("teamExportButtons");
  if (teamExportDiv) {
    const rosterSeasonQuery = encodeURIComponent(rosterSeason);
    const statsSeasonQuery  = encodeURIComponent(_statsSeason);
    teamExportDiv.innerHTML = `
      <button class="btn btn-sm" onclick="window.location.href='/export/2k/team/${abbr}?season=${statsSeasonQuery}&roster_season=${rosterSeasonQuery}'">📦 Export Team 2K JSON ZIP</button>
      <button class="btn btn-sm" onclick="window.location.href='/export/excel/team/${abbr}?season=${statsSeasonQuery}&roster_season=${rosterSeasonQuery}'">📥 Export Team Excel</button>
      <button class="btn btn-sm" onclick="window.location.href='/export/excel/team/${abbr}/attributes?season=${statsSeasonQuery}&roster_season=${rosterSeasonQuery}'">📥 Export Attributes Excel</button>
    `;
  }

  // Phase 2: Render team roster card grid
  renderTeamRosterGrid(data, season, rosterSeason);

  // Store team data for JSON copy
  window._teamData = data;
}

function renderTeamRosterGrid(data, season, rosterSeason) {
  // Reset expanded view
  teamExpandedPlayerDetail.hidden = true;
  
  // Render card grid
  teamRosterGrid.innerHTML = data.players.map((player, idx) => {
    const allAttrs = Object.values(player.attributes || {});
    const ovrValue = allAttrs.length > 0
      ? Math.round(allAttrs.reduce((sum, a) => sum + a.value, 0) / allAttrs.length)
      : 75;
    
    const grades = calculateSummaryGrades(player.tendencies || {}, player.attributes || {});
    const encodedName = encodeURIComponent(player.player_name);
    
    return `
      <div class="team-roster-card" onclick="expandTeamPlayer(${idx}, event)">
        <div class="team-roster-card-header">
          <div>
            <div class="team-roster-card-name">${player.player_name}</div>
            <div class="team-roster-card-meta">${player.position || ""}</div>
          </div>
          <div class="team-roster-card-ovr">${ovrValue}</div>
        </div>
        
        <div class="team-roster-card-summaries">
          <div class="team-roster-summary-badge">
            <span class="team-roster-summary-label">OFF</span>
            <span class="team-roster-summary-grade">${grades.offense}</span>
          </div>
          <div class="team-roster-summary-badge">
            <span class="team-roster-summary-label">DEF</span>
            <span class="team-roster-summary-grade">${grades.defense}</span>
          </div>
          <div class="team-roster-summary-badge">
            <span class="team-roster-summary-label">PHY</span>
            <span class="team-roster-summary-grade">${grades.physicals}</span>
          </div>
        </div>
        
        <div class="team-roster-card-actions">
          <button class="btn btn-sm" onclick="window.location.href='/export/2k/${encodedName}?season=${season}'; event.stopPropagation();">2K JSON</button>
          <button class="btn btn-sm" onclick="window.location.href='/export/excel/${encodedName}?season=${season}'; event.stopPropagation();">Excel</button>
        </div>
      </div>`;
  }).join("");
  
  // Store players data for expanded view
  window._teamRosterData = data.players;
  window._teamSeason = season;
  window._teamRosterSeason = rosterSeason;
}

function expandTeamPlayer(playerIdx, event) {
  event.stopPropagation();
  const player = window._teamRosterData[playerIdx];
  if (!player) return;
  
  // Build the expanded profile content
  const allAttrs = Object.values(player.attributes || {});
  const ovrValue = allAttrs.length > 0
    ? Math.round(allAttrs.reduce((sum, a) => sum + a.value, 0) / allAttrs.length)
    : 75;
  
  const grades = calculateSummaryGrades(player.tendencies || {}, player.attributes || {});
  
  // Build enriched tendencies
  const enrichedTendencies = {};
  for (const [key, entry] of Object.entries(player.tendencies || {})) {
    enrichedTendencies[key] = {
      label: entry.label,
      value: entry.value,
      category: guessCategoryFromKey(key),
    };
  }
  
  // Render profile header
  const headerHtml = `
    <div class="profile-card-header">
      <div class="profile-card-info">
        <div class="profile-card-name">${player.player_name}</div>
        <div class="profile-card-meta">${[player.position, player.team, window._teamSeason].filter(Boolean).join(" · ")}</div>
      </div>
      <div class="profile-card-ovr">${ovrValue}</div>
    </div>
    
    <div class="profile-card-summaries">
      <div class="summary-badge">
        <div class="summary-badge-label">Offense</div>
        <div class="summary-badge-grade">${grades.offense}</div>
      </div>
      <div class="summary-badge">
        <div class="summary-badge-label">Defense</div>
        <div class="summary-badge-grade">${grades.defense}</div>
      </div>
      <div class="summary-badge">
        <div class="summary-badge-label">Physicals</div>
        <div class="summary-badge-grade">${grades.physicals}</div>
      </div>
    </div>
    
    <div class="profile-card-stats">
      <div class="stat-item">
        <span class="stat-label">GPG</span>
        <span class="stat-value">${player.gp || "N/A"}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">MIN</span>
        <span class="stat-value">${player.mpg ? player.mpg.toFixed(1) : "N/A"}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">PPG</span>
        <span class="stat-value">${player.ppg ? player.ppg.toFixed(1) : "N/A"}</span>
      </div>
    </div>
    
    <div class="profile-card-actions">
      <button class="btn btn-sm btn-action" id="teamExpandTendenciesBtn">View Tendencies</button>
      <button class="btn btn-sm btn-action" id="teamExpandAttributesBtn">View Attributes</button>
    </div>
    
    <div id="teamExpandTendenciesPanel" class="collapsible-panel" hidden>
      <div class="collapsible-header">
        <span class="collapsible-title">Tendencies</span>
        <span class="collapsible-toggle">▼</span>
      </div>
      <div class="collapsible-content">${renderTendenciesToHtml(enrichedTendencies)}</div>
    </div>
    
    <div id="teamExpandAttributesPanel" class="collapsible-panel" hidden>
      <div class="collapsible-header">
        <span class="collapsible-title">Attributes</span>
        <span class="collapsible-toggle">▼</span>
      </div>
      <div class="collapsible-content">${renderAttributesToHtml(player.attributes || {})}</div>
    </div>
  `;
  
  teamExpandedContent.innerHTML = headerHtml;
  teamExpandedPlayerDetail.hidden = false;
  
  // Scroll into view
  window.scrollTo(0, 0);
  
  // Set up collapsible headers
  const expandHeaders = teamExpandedContent.querySelectorAll(".collapsible-header");
  expandHeaders.forEach(header => {
    header.addEventListener("click", () => {
      const panel = header.closest(".collapsible-panel");
      panel.classList.toggle("open");
    });
  });
  
  // Set up action buttons
  const tendenciesBtn = document.getElementById("teamExpandTendenciesBtn");
  const attributesBtn = document.getElementById("teamExpandAttributesBtn");
  const tendenciesPanel = document.getElementById("teamExpandTendenciesPanel");
  const attributesPanel = document.getElementById("teamExpandAttributesPanel");
  
  if (tendenciesBtn) tendenciesBtn.addEventListener("click", () => {
    tendenciesPanel.classList.remove("hidden");
    tendenciesPanel.classList.add("open");
    attributesPanel.classList.add("hidden");
    attributesPanel.classList.remove("open");
  });
  
  if (attributesBtn) attributesBtn.addEventListener("click", () => {
    attributesPanel.classList.remove("hidden");
    attributesPanel.classList.add("open");
    tendenciesPanel.classList.add("hidden");
    tendenciesPanel.classList.remove("open");
  });
}

// Close expanded player detail
teamCloseExpandedBtn.addEventListener("click", () => {
  teamExpandedPlayerDetail.hidden = true;
  document.getElementById("teamRosterGrid").scrollIntoView({ behavior: "smooth" });
});


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

async function loadCurrentUser() {
  if (!_authToken) {
    _authUser = null;
    applyAuthState();
    if (window.location.pathname === "/app") {
      window.location.replace("/");
    }
    return;
  }

  try {
    const resp = await apiFetch("/auth/me");
    if (!resp.ok) {
      _authUser = null;
      _authToken = "";
      localStorage.removeItem("authToken");
      applyAuthState();
      if (window.location.pathname === "/app") {
        window.location.replace("/");
      }
      return;
    }
    _authUser = await resp.json();
    if (_authUser && _authUser.must_change_password && window.location.pathname === "/app") {
      window.location.replace("/");
      return;
    }
    applyAuthState();
  } catch {
    _authUser = null;
    _authToken = "";
    localStorage.removeItem("authToken");
    applyAuthState();
    if (window.location.pathname === "/app") {
      window.location.replace("/");
    }
  }
}

function logout() {
  _authToken = "";
  _authUser = null;
  localStorage.removeItem("authToken");
  window.location.replace("/");
}

function collectChangedCorrections() {
  const changed = {};
  tendenciesContainer.querySelectorAll(".tendency-edit").forEach(inputEl => {
    const key = inputEl.dataset.key;
    const base = Number(_tendencyOriginalValues[key] || 0);
    const now = Number(inputEl.value || 0);
    if (!Number.isFinite(now)) return;
    if (now !== base) {
      changed[key] = Math.max(0, Math.min(100, Math.round(now)));
    }
  });
  return changed;
}

async function saveEditorFeedback() {
  if (!canEditTendencies()) {
    showError("Only editor/admin accounts can submit corrections.");
    return;
  }
  if (!_currentPlayerData || !_currentPlayerData.player_id) {
    showError("Generate a player first.");
    return;
  }

  const corrections = collectChangedCorrections();
  const keys = Object.keys(corrections);
  if (!keys.length) {
    editorFeedbackStatus.textContent = "No changes detected.";
    return;
  }

  editorFeedbackStatus.textContent = "Saving...";
  const resp = await apiFetch("/feedback/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      player_id: _currentPlayerData.player_id,
      corrections,
      notes: (editorFeedbackNotes.value || "").trim(),
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    editorFeedbackStatus.textContent = "Save failed";
    showError(`Feedback save failed: ${err.detail || "Unknown error"}`);
    return;
  }

  const data = await resp.json();
  editorFeedbackStatus.textContent = `Saved ${data.saved_count || keys.length} correction(s).`;
  hideError();

  const currentName = _currentPlayerData.player_name;
  if (currentName) {
    await generatePlayer(currentName);
  }
}

logoutBtn.addEventListener("click", logout);
saveFeedbackBtn.addEventListener("click", saveEditorFeedback);
if (adminUsersBtn) {
  adminUsersBtn.addEventListener("click", () => {
    window.location.href = "/admin";
  });
}

populateSeasonOptions();
updateDataModeBadge(seasonSelect.value);
seasonSelect.addEventListener("change", () => {
  updateDataModeBadge(seasonSelect.value);
});

loadCurrentUser();
