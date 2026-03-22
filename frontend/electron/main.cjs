const { app, BrowserWindow, dialog } = require("electron");
const fs = require("fs");
const path = require("path");

function readDesktopConfig() {
  const configPath = path.join(__dirname, "..", "desktop.config.json");
  try {
    const raw = fs.readFileSync(configPath, "utf-8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function resolveTargetUrl() {
  const cliOverride = process.argv.find((arg) => arg.startsWith("--app-url="));
  if (cliOverride) return cliOverride.replace("--app-url=", "").trim();

  const envOverride = String(process.env.N2K_DESKTOP_APP_URL || "").trim();
  if (envOverride) return envOverride;

  const cfg = readDesktopConfig();
  const cfgUrl = typeof cfg.appUrl === "string" ? cfg.appUrl.trim() : "";
  if (cfgUrl) return cfgUrl;

  return "http://localhost:3000";
}

function createMainWindow() {
  const targetUrl = resolveTargetUrl();
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    title: "ATD N2K26 Generator",
    backgroundColor: "#0b1220",
    webPreferences: {
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
    },
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http:")) return { action: "allow" };
    if (url.startsWith("https:")) return { action: "allow" };
    return { action: "deny" };
  });

  win.webContents.on("did-fail-load", (_event, errorCode, errorDescription) => {
    dialog.showErrorBox(
      "Unable to open app",
      `Desktop app could not load ${targetUrl}\n\nError ${errorCode}: ${errorDescription}`,
    );
  });

  win.loadURL(targetUrl);
}

app.whenReady().then(() => {
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
