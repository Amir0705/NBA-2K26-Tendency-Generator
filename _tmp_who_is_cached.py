import sqlite3, json
conn = sqlite3.connect('data/cache/nba_cache.db')
row = conn.execute('SELECT value FROM cache WHERE key=?', ('all_players',)).fetchone()
if not row:
    print('No all_players cache')
else:
    players = json.loads(row[0])
    ids = {1626156,1627742,1627936,1628369,1628401,1628404,1628415,1628973,1629012,1629029,
           1629645,1629750,1630162,1630183,1630228,1630264,1630596,1630643,1631099,1631120,
           1631128,201142,201935,201950,202695,203110,203468,203507,203952,203954,203994}
    for p in players:
        if p.get('PERSON_ID') in ids:
            print(f"{p['PERSON_ID']}: {p.get('DISPLAY_FIRST_LAST')} | {p.get('TEAM_ABBREVIATION','?')}")

# Also print player_info positions for cached IDs
print()
print("-- Positions from player_info cache --")
for pid in sorted(ids):
    row2 = conn.execute('SELECT value FROM cache WHERE key=?', (f'player_info:{pid}',)).fetchone()
    if row2:
        info = json.loads(row2[0])
        print(f"  {pid}: pos={info.get('position','?')}  ht={info.get('height','?')}  wt={info.get('weight','?')}")
