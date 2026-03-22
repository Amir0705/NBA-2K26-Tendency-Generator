CREATE OR REPLACE VIEW player_generation_source_v1 AS
WITH stats_union AS (
  SELECT
    p.player_id,
    p.season AS season_label,
    split_part(p.season, '-', 1)::int AS season_start,
    1 AS source_priority,
    p.gp::real AS gp,
    p.pts_pg::real AS pts_pg,
    p.reb_pg::real AS reb_pg,
    p.ast_pg::real AS ast_pg,
    p.stl_pg::real AS stl_pg,
    p.blk_pg::real AS blk_pg,
    p.fg_pct::real AS fg_pct,
    p.fg3_pct::real AS fg3_pct
  FROM pbp_profiles p

  UNION ALL

  SELECT
    s.player_id,
    s.season AS season_label,
    split_part(s.season, '-', 1)::int AS season_start,
    2 AS source_priority,
    s.gp::real AS gp,
    s.pts_pg::real AS pts_pg,
    s.reb_pg::real AS reb_pg,
    s.ast_pg::real AS ast_pg,
    s.stl_pg::real AS stl_pg,
    s.blk_pg::real AS blk_pg,
    s.fg_pct::real AS fg_pct,
    s.fg3_pct::real AS fg3_pct
  FROM player_seasons s
),
stats_dedup AS (
  SELECT *
  FROM (
    SELECT
      u.*,
      row_number() OVER (
        PARTITION BY u.player_id, u.season_start
        ORDER BY u.source_priority
      ) AS rn
    FROM stats_union u
  ) x
  WHERE x.rn = 1
),
stats_enriched AS (
  SELECT
    d.*,
    lag(d.gp)      OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_gp,
    lag(d.pts_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_pts_pg,
    lag(d.reb_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_reb_pg,
    lag(d.ast_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_ast_pg,
    lag(d.stl_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_stl_pg,
    lag(d.blk_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_blk_pg,
    lag(d.fg_pct)  OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_fg_pct,
    lag(d.fg3_pct) OVER (PARTITION BY d.player_id ORDER BY d.season_start) AS prev_fg3_pct,

    avg(d.gp)      OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_gp,
    avg(d.pts_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_pts_pg,
    avg(d.reb_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_reb_pg,
    avg(d.ast_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_ast_pg,
    avg(d.stl_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_stl_pg,
    avg(d.blk_pg)  OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_blk_pg,
    avg(d.fg_pct)  OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_fg_pct,
    avg(d.fg3_pct) OVER (PARTITION BY d.player_id ORDER BY d.season_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS career_fg3_pct
  FROM stats_dedup d
)
SELECT
  e.player_id,
  e.season_label,
  e.season_start,

  coalesce(tr.full_name, pi.full_name) AS full_name,
  tr.team_abbr,
  coalesce(tr.position, pi.position) AS position,

  pi.height_in,
  pi.weight_lbs,
  pi.birthdate,

  e.gp,
  e.pts_pg,
  e.reb_pg,
  e.ast_pg,
  e.stl_pg,
  e.blk_pg,
  e.fg_pct,
  e.fg3_pct,

  e.prev_gp,
  e.prev_pts_pg,
  e.prev_reb_pg,
  e.prev_ast_pg,
  e.prev_stl_pg,
  e.prev_blk_pg,
  e.prev_fg_pct,
  e.prev_fg3_pct,

  e.career_gp,
  e.career_pts_pg,
  e.career_reb_pg,
  e.career_ast_pg,
  e.career_stl_pg,
  e.career_blk_pg,
  e.career_fg_pct,
  e.career_fg3_pct,

  pi.age,
  pi.years_pro,
  pi.draft,
  pi.school

FROM stats_enriched e
LEFT JOIN team_rosters tr
  ON tr.player_id = e.player_id
 AND split_part(tr.season, '-', 1)::int = e.season_start
LEFT JOIN player_info pi
  ON pi.player_id = e.player_id;

-- Quick validation query:
-- SELECT *
-- FROM player_generation_source_v1
-- WHERE player_id = 1629029
-- ORDER BY season_start DESC
-- LIMIT 3;
