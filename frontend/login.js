const loginUsername = document.getElementById("loginUsername");
const loginPassword = document.getElementById("loginPassword");
const loginBtn = document.getElementById("loginBtn");
const mustChangePanel = document.getElementById("mustChangePanel");
const loginPanel = document.getElementById("loginPanel");
const newPassword = document.getElementById("newPassword");
const changePasswordBtn = document.getElementById("changePasswordBtn");
const loginError = document.getElementById("loginError");

let authToken = localStorage.getItem("authToken") || "";

function showError(msg) {
  loginError.textContent = msg;
  loginError.hidden = false;
}

function hideError() {
  loginError.hidden = true;
}

async function authFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`;
  }
  return fetch(url, { ...options, headers });
}

async function redirectIfAlreadyAuthenticated() {
  if (!authToken) return;

  try {
    const resp = await authFetch("/auth/me");
    if (!resp.ok) {
      localStorage.removeItem("authToken");
      authToken = "";
      return;
    }

    const me = await resp.json();
    if (me.must_change_password) {
      loginPanel.hidden = true;
      mustChangePanel.hidden = false;
      return;
    }

    window.location.replace("/app");
  } catch {
    localStorage.removeItem("authToken");
    authToken = "";
  }
}

async function login() {
  const username = (loginUsername.value || "").trim();
  const password = loginPassword.value || "";
  if (!username || !password) {
    showError("Please enter username and password.");
    return;
  }

  hideError();
  const resp = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    showError(err.detail || "Login failed");
    return;
  }

  const data = await resp.json();
  authToken = data.token;
  localStorage.setItem("authToken", authToken);
  loginPassword.value = "";

  if (data.must_change_password) {
    loginPanel.hidden = true;
    mustChangePanel.hidden = false;
    return;
  }

  window.location.replace("/app");
}

async function changePassword() {
  const nextPassword = (newPassword.value || "").trim();
  if (nextPassword.length < 4) {
    showError("New password must be at least 4 characters.");
    return;
  }

  hideError();
  const resp = await authFetch("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_password: nextPassword }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    showError(err.detail || "Failed to change password");
    return;
  }

  window.location.replace("/app");
}

loginBtn.addEventListener("click", login);
loginPassword.addEventListener("keydown", (e) => {
  if (e.key === "Enter") login();
});
changePasswordBtn.addEventListener("click", changePassword);

redirectIfAlreadyAuthenticated();
