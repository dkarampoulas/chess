"""
download_data.py
Download historical 10-minute (600+0) rapid chess matches from Lichess.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "rapid_10min_matches.jsonl")

def fetch_json(url, headers=None):
    if headers is None:
        headers = {'Accept': 'application/json'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get_seed_users(target_count=80):
    users = []
    # 1. Top rapid players
    try:
        data = fetch_json("https://lichess.org/api/player/top/100/rapid")
        for u in data.get('users', []):
            users.append(u['username'])
    except Exception as e:
        print(f"Error fetching top rapid players: {e}")

    # Fallback/popular active usernames if needed
    fallbacks = [
        "Unicorn7Love", "espaguetis", "KalleThicc", "Andrey11976", "Night-King96",
        "Penguingim1", "MagnusCarlsen", "Hikaru", "nihalsarin2004", "DanielNaroditsky",
        "RebeccaHarris", "alireza2003", "FabianoCaruana", "LyonBeast", "ChessNetwork",
        "EricRosen", "GMWSO", "fairchess_on_youtube", "Polish_fighter3000", "Dr_Drunkenstein"
    ]
    for fb in fallbacks:
        if fb not in users:
            users.append(fb)
    return users[:target_count]

def download_rapid_10m_matches(target_games=3000):
    print(f"Starting download of 10-minute (600+0) rapid matches. Target: {target_games} games...")
    seed_users = get_seed_users()
    print(f"Discovered {len(seed_users)} seed players.")

    collected_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        g = json.loads(line)
                        collected_ids.add(g["id"])
                    except Exception:
                        pass
        print(f"Loaded {len(collected_ids)} already existing games from {OUTPUT_FILE}.")

    if len(collected_ids) >= target_games:
        print(f"Target already met ({len(collected_ids)} >= {target_games}). Done!")
        return

    visited_users = set()
    user_queue = list(seed_users)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        while user_queue and len(collected_ids) < target_games:
            current_user = user_queue.pop(0)
            if current_user in visited_users:
                continue
            visited_users.add(current_user)

            url = f"https://lichess.org/api/games/user/{current_user}?perfType=rapid&max=60&clocks=true&opening=true"
            req = urllib.request.Request(url, headers={'Accept': 'application/x-ndjson'})
            
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    lines = resp.read().decode('utf-8').strip().split('\n')
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print("Hit rate limit (HTTP 429). Sleeping 30 seconds...")
                    time.sleep(30)
                    user_queue.append(current_user)
                    continue
                else:
                    print(f"HTTP error for {current_user}: {e.code}")
                    continue
            except Exception as e:
                print(f"Connection error for {current_user}: {e}")
                continue

            user_added = 0
            for line in lines:
                if not line.strip():
                    continue
                try:
                    g = json.loads(line)
                except Exception:
                    continue

                gid = g.get("id")
                if not gid or gid in collected_ids:
                    continue

                clock = g.get("clock")
                # Strictly 10-minute rapid: initial=600 seconds, increment=0
                if not clock or clock.get("initial") != 600 or clock.get("increment") != 0:
                    continue

                # Ensure game has moves and time info
                moves = g.get("moves")
                if not moves or len(moves.split()) < 4:
                    continue

                created_at = g.get("createdAt")
                last_move_at = g.get("lastMoveAt")
                if not created_at or not last_move_at or last_move_at <= created_at:
                    continue

                # Add opponent to queue to ensure rich network diversity
                white_player = g.get("players", {}).get("white", {}).get("user", {}).get("name")
                black_player = g.get("players", {}).get("black", {}).get("user", {}).get("name")
                for p in [white_player, black_player]:
                    if p and p not in visited_users and len(user_queue) < 300:
                        user_queue.append(p)

                out_f.write(json.dumps(g) + "\n")
                out_f.flush()
                collected_ids.add(gid)
                user_added += 1

                if len(collected_ids) >= target_games:
                    break

            print(f"[{len(collected_ids)}/{target_games}] Processed {current_user}: added {user_added} 10+0 games. (Queue: {len(user_queue)})")
            time.sleep(0.5)

    print(f"Download complete! Total 10-minute rapid games saved: {len(collected_ids)} in {OUTPUT_FILE}")

if __name__ == "__main__":
    download_rapid_10m_matches(target_games=2500)
