# NBA-2K26-Tendency-Generator

Reference docs:

- `ATTRIBUTE_GLOSSARY.md`: Gameplay and simulation impact of core 2K attributes.

## Firebase Deployment (Render)

This project can run in two modes for auth and feedback persistence:

- Local JSON mode (default)
- Firebase mode (recommended for production)

### Enable Firebase mode

Set the following environment variables on Render:

- `N2K_USE_FIREBASE=1`
- `FIREBASE_API_KEY=<firebase web api key>`
- `FIREBASE_PROJECT_ID=<your firebase project id>`
- One of:
	- `FIREBASE_SERVICE_ACCOUNT_JSON=<full service account json string>`
	- `FIREBASE_SERVICE_ACCOUNT_FILE=<path to service account json file>`

Optional admin bootstrap variables:

- `N2K_ADMIN_USERNAME` (default: `admin`)
- `N2K_ADMIN_PASSWORD` (default: `admin123`)
- `N2K_ADMIN_FULL_NAME` (default: `Administrator`)

### Optional PBP Stats context features

You can augment tendency generation with possession-level and shot-context signals
from PBP Stats. This is optional and automatically falls back to nba_api-only
features if disabled or unavailable.

Set:

- `N2K_USE_PBPSTATS=1`
- `PBPSTATS_API_KEY=<optional api key>`
- `PBPSTATS_BASE_URL=https://api.pbpstats.com` (optional override)

### PBP-only generation mode

The generator now supports a PBP-only path for feature construction and player
search (no nba_api dependency in the generation flow).

Set:

- `N2K_PBP_ONLY=1` (default)

When enabled, player lookup and tendency features are derived from PBP Stats
endpoints (`get-all-players-for-league`, `get-totals`, `get-shot-query-summary`).

### Notes

- In Firebase mode, user auth is handled by Firebase Authentication.
- User role/must-change-password metadata is stored in Firestore collection `app_users`.
- Editor feedback is stored in Firestore collection `feedback_entries`.
- If Firebase init fails, the app automatically falls back to local JSON stores.