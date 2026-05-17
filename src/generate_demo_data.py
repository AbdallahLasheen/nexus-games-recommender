"""
Demo Data Generator — AIE425 IRS
==================================
Generates realistic synthetic Video Games data so the full pipeline
(preprocessing → training → Flask server) can run WITHOUT downloading
the Amazon dataset.

Run:  python src/generate_demo_data.py --output data/raw
"""

import os
import json
import random
import numpy as np
from pathlib import Path


GAMES = [
    ("B001", "The Witcher 3: Wild Hunt",       "action rpg open world fantasy",          59.99, ["Action", "RPG"]),
    ("B002", "Cyberpunk 2077",                  "action rpg futuristic open world sci-fi", 49.99, ["Action", "RPG"]),
    ("B003", "Elden Ring",                      "souls-like action rpg challenging",        59.99, ["Action", "RPG", "Souls-like"]),
    ("B004", "Horizon Zero Dawn",               "action adventure open world robot",        29.99, ["Action", "Adventure"]),
    ("B005", "God of War",                      "action adventure mythology combat",        39.99, ["Action", "Adventure"]),
    ("B006", "Red Dead Redemption 2",           "open world western adventure story",       39.99, ["Adventure", "Action"]),
    ("B007", "The Legend of Zelda: BotW",       "open world adventure exploration",         59.99, ["Adventure", "Action"]),
    ("B008", "Dark Souls III",                  "souls-like difficult action combat",        39.99, ["Action", "Souls-like"]),
    ("B009", "Minecraft",                       "sandbox survival building creative",        26.99, ["Sandbox", "Survival"]),
    ("B010", "Stardew Valley",                  "farming simulation relaxing indie",         14.99, ["Simulation", "Indie"]),
    ("B011", "Among Us",                        "social deduction multiplayer strategy",      4.99, ["Strategy", "Multiplayer"]),
    ("B012", "Hades",                           "roguelike action dungeon greek mythology",  24.99, ["Action", "Roguelike", "Indie"]),
    ("B013", "Celeste",                         "platformer difficult indie story",          19.99, ["Platformer", "Indie"]),
    ("B014", "Hollow Knight",                   "metroidvania action exploration dark",       14.99, ["Action", "Indie"]),
    ("B015", "Disco Elysium",                   "rpg detective story narrative",             39.99, ["RPG", "Adventure"]),
    ("B016", "Death Stranding",                 "action adventure post-apocalyptic",          39.99, ["Action", "Adventure"]),
    ("B017", "Sekiro: Shadows Die Twice",       "souls-like samurai action combat",          59.99, ["Action", "Souls-like"]),
    ("B018", "Monster Hunter: World",           "action rpg hunting multiplayer",            29.99, ["Action", "RPG"]),
    ("B019", "Divinity: Original Sin 2",        "turn-based rpg strategy fantasy",           44.99, ["RPG", "Strategy"]),
    ("B020", "Baldur's Gate 3",                 "turn-based rpg dnd fantasy story",          59.99, ["RPG", "Strategy"]),
    ("B021", "Persona 5 Royal",                 "jrpg social simulation story",             59.99, ["RPG", "JRPG"]),
    ("B022", "Final Fantasy XIV",               "mmorpg story fantasy online",              39.99, ["RPG", "MMO"]),
    ("B023", "Overwatch 2",                     "first person shooter multiplayer hero",       0.00, ["FPS", "Multiplayer"]),
    ("B024", "Apex Legends",                    "battle royale fps multiplayer free",          0.00, ["FPS", "Battle Royale"]),
    ("B025", "Fortnite",                        "battle royale building shooter free",         0.00, ["FPS", "Battle Royale"]),
    ("B026", "League of Legends",               "moba strategy multiplayer competitive",       0.00, ["MOBA", "Strategy"]),
    ("B027", "Dota 2",                          "moba strategy fantasy competitive",           0.00, ["MOBA", "Strategy"]),
    ("B028", "Counter-Strike 2",                "tactical fps competitive shooter",            0.00, ["FPS", "Strategy"]),
    ("B029", "Valorant",                        "tactical fps hero shooter competitive",       0.00, ["FPS", "Strategy"]),
    ("B030", "Hearthstone",                     "card game strategy fantasy digital",          0.00, ["Strategy", "Card Game"]),
    ("B031", "Slay the Spire",                  "roguelike card game strategy indie",         24.99, ["Strategy", "Roguelike"]),
    ("B032", "Into the Breach",                 "turn-based strategy sci-fi mech",            14.99, ["Strategy", "Indie"]),
    ("B033", "Civilization VI",                 "4x strategy historical turn-based",           29.99, ["Strategy"]),
    ("B034", "XCOM 2",                          "turn-based strategy aliens tactical",         29.99, ["Strategy"]),
    ("B035", "Portal 2",                        "puzzle platformer physics co-op",             9.99, ["Puzzle", "Platformer"]),
    ("B036", "The Talos Principle",             "puzzle philosophical first person",           19.99, ["Puzzle"]),
    ("B037", "Outer Wilds",                     "exploration mystery space indie",             24.99, ["Adventure", "Indie"]),
    ("B038", "Subnautica",                      "survival exploration underwater sci-fi",      29.99, ["Survival", "Adventure"]),
    ("B039", "Terraria",                        "sandbox survival 2d exploration",             9.99, ["Sandbox", "Action"]),
    ("B040", "No Man's Sky",                    "space exploration survival sandbox",          59.99, ["Adventure", "Survival"]),
    ("B041", "Mass Effect Legendary",           "action rpg sci-fi story trilogy",             59.99, ["RPG", "Action"]),
    ("B042", "Dragon Age: Inquisition",         "action rpg fantasy story party",             19.99, ["RPG", "Action"]),
    ("B043", "Pathfinder: Wrath of Righteous",  "turn-based rpg fantasy crpg",                39.99, ["RPG", "Strategy"]),
    ("B044", "Pillars of Eternity II",          "isometric rpg naval fantasy story",           24.99, ["RPG"]),
    ("B045", "Torment: Tides of Numenera",      "rpg narrative story philosophical",           19.99, ["RPG", "Adventure"]),
    ("B046", "Ghost of Tsushima",               "action adventure samurai open world",         49.99, ["Action", "Adventure"]),
    ("B047", "Nioh 2",                          "souls-like samurai action loot",              49.99, ["Action", "Souls-like"]),
    ("B048", "Star Wars Jedi: Fallen Order",    "action adventure star wars story",            39.99, ["Action", "Adventure"]),
    ("B049", "Returnal",                        "roguelike third person shooter sci-fi",       59.99, ["Action", "Roguelike"]),
    ("B050", "Deathloop",                       "fps roguelike immersive sim",                 39.99, ["FPS", "Action"]),
]

USERS = [f"U{i:04d}" for i in range(1, 201)]    # 200 users


def generate_user_preferences(seed: int) -> dict:
    """Each user has random genre preferences."""
    rng = random.Random(seed)
    all_genres = list({g for _, _, _, _, genres in GAMES for g in genres})
    preferred = rng.sample(all_genres, k=rng.randint(2, 5))
    return preferred


def simulate_rating(asin: str, genres: list, user_prefs: list, rng: random.Random) -> float:
    """Simulate a rating based on genre overlap."""
    game = next((g for g in GAMES if g[0] == asin), None)
    if game is None:
        return rng.uniform(1, 5)
    overlap = len(set(game[4]) & set(user_prefs))
    base    = 2.5 + overlap * 0.7 + rng.uniform(-0.8, 0.8)
    return max(1.0, min(5.0, round(base * 2) / 2))   # round to 0.5


def generate_reviews(n_per_user: int = 15) -> list:
    records = []
    for uid in USERS:
        seed      = hash(uid) % 10_000
        rng       = random.Random(seed)
        prefs     = generate_user_preferences(seed)
        game_pool = rng.sample(GAMES, k=min(n_per_user + 5, len(GAMES)))
        rated     = rng.sample(game_pool, k=rng.randint(n_per_user, n_per_user + 5))
        for game in rated:
            asin   = game[0]
            rating = simulate_rating(asin, game[4], prefs, rng)
            records.append({
                "user_id":     uid,
                "parent_asin": asin,
                "rating":      rating,
                "timestamp":   1_700_000_000 + rng.randint(0, 5_000_000),
            })
    return records


def generate_meta() -> list:
    records = []
    for asin, title, desc, price, cats in GAMES:
        records.append({
            "parent_asin": asin,
            "title":       title,
            "description": [desc + " — A highly acclaimed video game with stunning visuals and deep gameplay mechanics."],
            "features":    [f"Genre: {', '.join(cats)}", f"Price: ${price:.2f}", "Platform: PC, Console"],
            "categories":  [cats],
            "price":       price if price > 0 else None,
        })
    return records


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw", help="Output directory for raw files")
    args = parser.parse_args()

    out = Path(args.output)
    print("Generating demo Reviews.json ...")
    reviews = generate_reviews(n_per_user=15)
    save_jsonl(reviews, out / "Reviews.json")
    print(f"  {len(reviews)} review records saved.")

    print("Generating demo Meta.json ...")
    meta = generate_meta()
    save_jsonl(meta, out / "Meta.json")
    print(f"  {len(meta)} meta records saved.")

    print(f"\n✅  Demo data saved to {out}/")
    print("  Next step:  python src/preprocess.py "
          "--reviews data/raw/Reviews.json --meta data/raw/Meta.json "
          "--output data/processed")
