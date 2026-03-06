"""Tests for NBAApiClient fallback and caching behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingest.nba_api_client import NBAApiClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> NBAApiClient:
    """Return a client with no disk cache."""
    return NBAApiClient(cache_dir=None)


STATIC_PLAYER_FIXTURE = [
    {"id": 1, "full_name": "Test Player", "is_active": True},
    {"id": 2, "full_name": "Inactive Guy", "is_active": False},
]

LIVE_PLAYER_FIXTURE = [
    {
        "PERSON_ID": 201939,
        "DISPLAY_FIRST_LAST": "Stephen Curry",
        "TEAM_ABBREVIATION": "GSW",
        "ROSTERSTATUS": 1,
    }
]


# ---------------------------------------------------------------------------
# search_player — fallback to static list when live API raises
# ---------------------------------------------------------------------------

class TestSearchPlayerFallback:
    def test_falls_back_when_get_all_players_raises(self):
        client = _make_client()
        with patch.object(client, "_get_all_players", side_effect=RuntimeError("API down")):
            with patch.object(client, "_get_static_players", return_value=LIVE_PLAYER_FIXTURE):
                results = client.search_player("curry")

        assert len(results) == 1
        assert results[0]["player_id"] == 201939
        assert results[0]["full_name"] == "Stephen Curry"

    def test_returns_empty_list_when_both_sources_have_no_match(self):
        client = _make_client()
        with patch.object(client, "_get_all_players", side_effect=RuntimeError("API down")):
            with patch.object(client, "_get_static_players", return_value=LIVE_PLAYER_FIXTURE):
                results = client.search_player("ZZZNOBODYZZ")

        assert results == []

    def test_uses_live_data_when_available(self):
        client = _make_client()
        with patch.object(client, "_get_all_players", return_value=LIVE_PLAYER_FIXTURE):
            results = client.search_player("curry")

        assert len(results) == 1
        assert results[0]["full_name"] == "Stephen Curry"


# ---------------------------------------------------------------------------
# _get_all_players — no caching of empty results
# ---------------------------------------------------------------------------

class TestGetAllPlayersEmptyCache:
    def _make_mock_endpoint(self, rows: list) -> MagicMock:
        """Build a mock nba_api endpoint object for the given rowSet."""
        endpoint = MagicMock()
        endpoint.get_dict.return_value = {
            "resultSets": [
                {
                    "headers": list(rows[0].keys()) if rows else ["PERSON_ID"],
                    "rowSet": [list(r.values()) for r in rows] if rows else [],
                }
            ]
        }
        return endpoint

    def test_does_not_cache_empty_result(self):
        client = _make_client()
        empty_endpoint = self._make_mock_endpoint([])
        with patch.object(client, "_with_retry", return_value=empty_endpoint):
            with patch.object(client, "_cache_set") as mock_cache_set:
                client._get_all_players()
                mock_cache_set.assert_not_called()

    def test_caches_non_empty_result(self):
        client = _make_client()
        row = {
            "PERSON_ID": 1,
            "DISPLAY_FIRST_LAST": "Player One",
            "TEAM_ABBREVIATION": "LAL",
            "ROSTERSTATUS": 1,
        }
        non_empty_endpoint = self._make_mock_endpoint([row])
        with patch.object(client, "_with_retry", return_value=non_empty_endpoint):
            with patch.object(client, "_cache_set") as mock_cache_set:
                client._get_all_players()
                mock_cache_set.assert_called_once()


# ---------------------------------------------------------------------------
# _get_static_players — format matches what search_player expects
# ---------------------------------------------------------------------------

class TestGetStaticPlayers:
    def test_returns_correct_keys(self):
        client = _make_client()
        with patch("nba_api.stats.static.players.get_players", return_value=STATIC_PLAYER_FIXTURE):
            result = client._get_static_players()

        assert len(result) == 2
        for record in result:
            assert "PERSON_ID" in record
            assert "DISPLAY_FIRST_LAST" in record
            assert "TEAM_ABBREVIATION" in record
            assert "ROSTERSTATUS" in record

    def test_active_flag_mapping(self):
        client = _make_client()
        with patch("nba_api.stats.static.players.get_players", return_value=STATIC_PLAYER_FIXTURE):
            result = client._get_static_players()

        active = next(r for r in result if r["PERSON_ID"] == 1)
        inactive = next(r for r in result if r["PERSON_ID"] == 2)
        assert active["ROSTERSTATUS"] == 1
        assert inactive["ROSTERSTATUS"] == 0

    def test_id_and_name_mapping(self):
        client = _make_client()
        with patch("nba_api.stats.static.players.get_players", return_value=STATIC_PLAYER_FIXTURE):
            result = client._get_static_players()

        assert result[0]["PERSON_ID"] == 1
        assert result[0]["DISPLAY_FIRST_LAST"] == "Test Player"
