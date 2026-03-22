# Desktop App Packaging

This project can be shipped as a portable Windows desktop app (no installer required).

## How it works

The desktop executable opens the hosted frontend URL in an Electron shell.
All existing logic remains the same:

- Login and auth flows in the Next.js frontend
- Player generation API calls and Supabase access
- Attribute/tendency logic currently used by your deployed services

## 1) Set the hosted app URL

Edit `desktop.config.json`:

```json
{
  "appUrl": "https://your-frontend-url.onrender.com"
}
```

## 2) Build portable app package

From the `frontend` folder:

```bash
npm install
npm run desktop:ship
```

Output:

- `dist-desktop/win-unpacked/` (portable app folder)
- `dist-desktop/ATD-N2K26-Generator-portable.zip` (share this file)

Share the zip file with your group. They unzip and run `ATD N2K26 Generator.exe`.

## Optional local dev mode

Run local Next + Electron together:

```bash
npm run desktop:dev
```

This opens `http://localhost:3000` by default.

## Optional runtime override

You can override URL while launching:

```bash
ATD-N2K26-Generator.exe --app-url=https://your-frontend-url.onrender.com
```

Or with env var:

```bash
set N2K_DESKTOP_APP_URL=https://your-frontend-url.onrender.com
ATD-N2K26-Generator.exe
```
