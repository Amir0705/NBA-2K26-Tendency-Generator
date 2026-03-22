-- Add demographics fields used by Player Info card.
-- Safe to run multiple times.

ALTER TABLE player_info
ADD COLUMN IF NOT EXISTS age integer;

ALTER TABLE player_info
ADD COLUMN IF NOT EXISTS years_pro integer;

ALTER TABLE player_info
ADD COLUMN IF NOT EXISTS draft text DEFAULT '';

ALTER TABLE player_info
ADD COLUMN IF NOT EXISTS school text DEFAULT '';

-- Optional: backfill age from birthdate for rows that have birthdate but no age.
UPDATE player_info
SET age = EXTRACT(YEAR FROM age(current_date, to_date(birthdate, 'YYYY-MM-DD')))
WHERE (age IS NULL OR age = 0)
  AND birthdate ~ '^\d{4}-\d{2}-\d{2}$';

-- Validation:
-- SELECT COUNT(*) AS total_players,
--        COUNT(*) FILTER (WHERE COALESCE(age, 0) > 0) AS with_age,
--        COUNT(*) FILTER (WHERE COALESCE(years_pro, 0) >= 0) AS with_years_pro,
--        COUNT(*) FILTER (WHERE COALESCE(draft, '') <> '') AS with_draft,
--        COUNT(*) FILTER (WHERE COALESCE(school, '') <> '') AS with_school
-- FROM player_info;
