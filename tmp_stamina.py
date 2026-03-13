import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.pipeline import TendencyPipeline
pipe = TendencyPipeline()
players = [
    (1629020, 'Vanderbilt'),
    (1629029, 'Luka'),
    (2544, 'LeBron'),
    (203999, 'Jokic'),
    (203507, 'Giannis'),
    (201935, 'Harden'),
    (201939, 'Curry'),
    (1630162, 'Edwards'),
    (203935, 'Smart'),
]
for pid, name in players:
    try:
        r = pipe.generate(pid, season='2024-25')
        f = r['features']
        a = r['attributes']
        print(name, " stamina=", a.get('stamina'), " min=", f.get('min_per_game',0), " gp=", f.get('games_played',0), " age=", f.get('age',0), " usg=", round(f.get('usage_rate',0),1))
    except Exception as e:
        print(name, "ERROR:", e)
