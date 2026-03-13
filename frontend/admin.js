const adminWelcome = document.getElementById("adminWelcome");
const adminRole = document.getElementById("adminRole");
const adminError = document.getElementById("adminError");
const createFullName = document.getElementById("createFullName");
const createUsername = document.getElementById("createUsername");
const createPassword = document.getElementById("createPassword");
const createRole = document.getElementById("createRole");
const createUserBtn = document.getElementById("createUserBtn");
const usersTableWrap = document.getElementById("usersTableWrap");
const openAppBtn = document.getElementById("openAppBtn");
const adminLogoutBtn = document.getElementById("adminLogoutBtn");

let authToken = localStorage.getItem("authToken") || "";
let me = null;

function showError(msg) {
  adminError.textContent = msg;
  adminError.hidden = false;
}

function hideError() {
  adminError.hidden = true;
}

async function apiFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`;
  }

  const response = await fetch(url, { ...options, headers });
  if (response.status === 401) {
    localStorage.removeItem("authToken");
    authToken = "";
    window.location.replace("/");
  }
  return response;
}

function logout() {
  localStorage.removeItem("authToken");
  authToken = "";
  window.location.replace("/");
}

function rowTemplate(user) {
  const mustChange = user.must_change_password ? "checked" : "";
  return `
    <tr>
      <td><input class="input table-input" data-edit="full_name" data-username="${user.username}" value="${user.full_name || ""}" /></td>
      <td><span class="user-chip">${user.username}</span></td>
      <td>
        <select class="input select table-input" data-edit="role" data-username="${user.username}">
          <option value="viewer" ${user.role === "viewer" ? "selected" : ""}>viewer</option>
          <option value="editor" ${user.role === "editor" ? "selected" : ""}>editor</option>
          <option value="admin" ${user.role === "admin" ? "selected" : ""}>admin</option>
        </select>
      </td>
      <td>
        <input type="checkbox" data-edit="must_change_password" data-username="${user.username}" ${mustChange} />
      </td>
      <td>
        <input type="password" class="input table-input" data-edit="reset_password" data-username="${user.username}" placeholder="Optional reset" />
      </td>
      <td class="users-actions">
        <button class="btn btn-sm" data-action="save" data-username="${user.username}">Save</button>
        <button class="btn btn-sm" data-action="delete" data-username="${user.username}">Delete</button>
      </td>
    </tr>
  `;
}

function renderUsers(users) {
  usersTableWrap.innerHTML = `
    <table class="users-table">
      <thead>
        <tr>
          <th>Full Name</th>
          <th>Username</th>
          <th>Role</th>
          <th>Must Change</th>
          <th>Reset Password</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${users.map(rowTemplate).join("")}
      </tbody>
    </table>
  `;

  usersTableWrap.querySelectorAll("button[data-action='save']").forEach((btn) => {
    btn.addEventListener("click", () => saveUser(btn.dataset.username));
  });

  usersTableWrap.querySelectorAll("button[data-action='delete']").forEach((btn) => {
    btn.addEventListener("click", () => deleteUser(btn.dataset.username));
  });
}

function valueFor(username, field) {
  const el = usersTableWrap.querySelector(`[data-edit='${field}'][data-username='${username}']`);
  if (!el) return null;
  if (el.type === "checkbox") return !!el.checked;
  return (el.value || "").trim();
}

async function refreshUsers() {
  const resp = await apiFetch("/auth/users");
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: "Failed to load users" }));
    showError(err.detail || "Failed to load users");
    return;
  }
  const data = await resp.json();
  renderUsers(Array.isArray(data.users) ? data.users : []);
}

async function createUser() {
  const payload = {
    full_name: (createFullName.value || "").trim(),
    username: (createUsername.value || "").trim(),
    password: (createPassword.value || "").trim(),
    role: (createRole.value || "viewer").trim().toLowerCase(),
  };

  if (!payload.username || !payload.password) {
    showError("Username and temporary password are required.");
    return;
  }

  const resp = await apiFetch("/auth/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: "Failed to create user" }));
    showError(err.detail || "Failed to create user");
    return;
  }

  createFullName.value = "";
  createUsername.value = "";
  createPassword.value = "";
  createRole.value = "viewer";
  hideError();
  await refreshUsers();
}

async function saveUser(username) {
  const payload = {
    full_name: valueFor(username, "full_name"),
    role: valueFor(username, "role"),
    must_change_password: valueFor(username, "must_change_password"),
  };

  const resetPassword = valueFor(username, "reset_password");
  if (resetPassword) {
    payload.reset_password = resetPassword;
    payload.must_change_password = true;
  }

  const resp = await apiFetch(`/auth/users/${encodeURIComponent(username)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: "Failed to update user" }));
    showError(err.detail || "Failed to update user");
    return;
  }

  hideError();
  await refreshUsers();
}

async function deleteUser(username) {
  if (!window.confirm(`Delete user '${username}'?`)) return;

  const resp = await apiFetch(`/auth/users/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: "Failed to delete user" }));
    showError(err.detail || "Failed to delete user");
    return;
  }

  hideError();
  await refreshUsers();
}

async function init() {
  if (!authToken) {
    window.location.replace("/");
    return;
  }

  const meResp = await apiFetch("/auth/me");
  if (!meResp.ok) return;
  me = await meResp.json();

  if (!me || me.role !== "admin") {
    window.location.replace("/app");
    return;
  }

  if (me.must_change_password) {
    window.location.replace("/");
    return;
  }

  adminWelcome.textContent = `Welcome, ${me.full_name || me.username}`;
  adminRole.textContent = `Role: ${me.role}`;
  hideError();
  await refreshUsers();
}

createUserBtn.addEventListener("click", createUser);
openAppBtn.addEventListener("click", () => window.location.href = "/app");
adminLogoutBtn.addEventListener("click", logout);

init();
