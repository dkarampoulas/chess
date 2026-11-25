"""
part2_analysis.py
Part 2: Spatial Piece Captures ("Deaths") Probability Theory & Tactical Brawl Analysis.
Calculates empirical death probability p_emp(i) for all 64 squares.
Analyzes top deadly squares, openings, move sequences, victim piece distributions,
brawl exchange dynamics, and overall battle winners.
"""

import json
import os
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import chess

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "rapid_10min_matches.jsonl")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "data", "part2_results.json")

PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King"
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def analyze_matches():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    # Square statistics
    # square_deaths[sq]: count of captures
    square_deaths = defaultdict(int)
    # square_piece_deaths[sq][piece_type]: count
    square_piece_deaths = defaultdict(lambda: defaultdict(int))
    # square_game_outcomes[sq]: { "white": count, "black": count, "draw": count }
    square_game_outcomes = defaultdict(lambda: Counter())

    # Deep-dive tracking for brawls
    # brawl_records: list of brawls across all squares
    brawl_records = defaultdict(list)
    # square_openings[sq]: Counter of (ECO, opening_name)
    square_openings = defaultdict(Counter)
    # square_movesets[sq]: Counter of initial move sequences (first 6 moves)
    square_movesets = defaultdict(Counter)

    total_games_analyzed = 0
    total_deaths = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue

            moves_str = g.get("moves")
            if not moves_str:
                continue

            winner = g.get("winner")
            if winner == "white":
                game_outcome = "white"
            elif winner == "black":
                game_outcome = "black"
            else:
                game_outcome = "draw"

            eco = g.get("opening", {}).get("eco", "Unknown")
            opening_name = g.get("opening", {}).get("name", "Unknown")

            moves = moves_str.split()
            board = chess.Board()

            game_squares_with_deaths = set()
            active_brawls = {}  # sq -> dict of current brawl state

            # Track first 6 moves (3 full turns) for opening moveset signature
            first_moves_sig = " ".join(moves[:6]) if len(moves) >= 6 else " ".join(moves)

            for ply, move_san in enumerate(moves):
                try:
                    move = board.parse_san(move_san)
                except Exception:
                    break

                is_capture = board.is_capture(move)
                if is_capture:
                    # Identify victim piece and death square
                    if board.is_en_passant(move):
                        victim_piece_type = chess.PAWN
                        victim_color = not board.turn
                        victim_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                        death_square_name = chess.square_name(victim_sq)
                    else:
                        victim = board.piece_at(move.to_square)
                        if victim:
                            victim_piece_type = victim.piece_type
                            victim_color = victim.color
                        else:
                            victim_piece_type = chess.PAWN
                            victim_color = not board.turn
                        death_square_name = chess.square_name(move.to_square)

                    capturing_piece = board.piece_at(move.from_square)
                    capturing_piece_type = capturing_piece.piece_type if capturing_piece else chess.PAWN
                    capturing_color = board.turn

                    square_deaths[death_square_name] += 1
                    total_deaths += 1
                    square_piece_deaths[death_square_name][victim_piece_type] += 1
                    game_squares_with_deaths.add(death_square_name)
                    square_openings[death_square_name][(eco, opening_name)] += 1
                    square_movesets[death_square_name][first_moves_sig] += 1

                    # Brawl tracking on death_square_name
                    # A brawl is a sequence of captures on the same square occurring within 3 plies of each other
                    sq_key = death_square_name
                    if sq_key in active_brawls:
                        brawl = active_brawls[sq_key]
                        if ply - brawl["last_ply"] <= 2:
                            # Continuation of brawl!
                            brawl["captures"].append({
                                "ply": ply,
                                "captor_color": "white" if capturing_color == chess.WHITE else "black",
                                "captor_piece": PIECE_NAMES.get(capturing_piece_type, "Pawn"),
                                "victim_color": "white" if victim_color == chess.WHITE else "black",
                                "victim_piece": PIECE_NAMES.get(victim_piece_type, "Pawn"),
                                "victim_val": PIECE_VALUES.get(victim_piece_type, 1)
                            })
                            brawl["last_ply"] = ply
                        else:
                            # Previous brawl ended; finalize and start new one
                            finalize_brawl(brawl, brawl_records, game_outcome, eco, opening_name)
                            active_brawls[sq_key] = init_brawl(sq_key, ply, capturing_color, capturing_piece_type, victim_color, victim_piece_type)
                    else:
                        active_brawls[sq_key] = init_brawl(sq_key, ply, capturing_color, capturing_piece_type, victim_color, victim_piece_type)

                board.push(move)

            # Finalize any remaining active brawls in this game
            for sq_key, brawl in active_brawls.items():
                finalize_brawl(brawl, brawl_records, game_outcome, eco, opening_name)

            for sq in game_squares_with_deaths:
                square_game_outcomes[sq][game_outcome] += 1

            total_games_analyzed += 1

    print(f"Analyzed {total_games_analyzed} games. Total pieces captured ('deaths'): {total_deaths}")
    return square_deaths, total_deaths, square_piece_deaths, square_game_outcomes, brawl_records, square_openings, square_movesets, total_games_analyzed

def init_brawl(sq, ply, capturing_color, capturing_piece_type, victim_color, victim_piece_type):
    return {
        "square": sq,
        "start_ply": ply,
        "last_ply": ply,
        "captures": [{
            "ply": ply,
            "captor_color": "white" if capturing_color == chess.WHITE else "black",
            "captor_piece": PIECE_NAMES.get(capturing_piece_type, "Pawn"),
            "victim_color": "white" if victim_color == chess.WHITE else "black",
            "victim_piece": PIECE_NAMES.get(victim_piece_type, "Pawn"),
            "victim_val": PIECE_VALUES.get(victim_piece_type, 1)
        }]
    }

def finalize_brawl(brawl, brawl_records, game_outcome, eco, opening_name):
    # Only record multi-capture brawls (2 or more captures on the same square in sequence)
    if len(brawl["captures"]) >= 2:
        # Calculate net material score for White
        # White gains value of Black pieces captured, loses value of White pieces captured
        white_material_gain = 0
        black_material_gain = 0
        for cap in brawl["captures"]:
            if cap["captor_color"] == "white":
                white_material_gain += cap["victim_val"]
            else:
                black_material_gain += cap["victim_val"]

        net_white_score = white_material_gain - black_material_gain
        if net_white_score > 0:
            brawl_winner = "white"
        elif net_white_score < 0:
            brawl_winner = "black"
        else:
            brawl_winner = "equal"

        brawl_records[brawl["square"]].append({
            "num_captures": len(brawl["captures"]),
            "captures": brawl["captures"],
            "brawl_winner": brawl_winner,
            "net_white_score": net_white_score,
            "game_outcome": game_outcome,
            "eco": eco,
            "opening_name": opening_name
        })

def run_part2():
    (square_deaths, total_deaths, square_piece_deaths,
     square_game_outcomes, brawl_records, square_openings,
     square_movesets, total_games) = analyze_matches()

    # Compile 64 square statistics
    squares_list = []
    ranks = "12345678"
    files = "abcdefgh"
    all_64_squares = [f + r for r in ranks for f in files]

    for sq in all_64_squares:
        d_count = square_deaths[sq]
        p_emp = d_count / total_deaths if total_deaths > 0 else 0.0

        # Piece distribution
        piece_breakdown = {}
        for p_code, p_name in PIECE_NAMES.items():
            if p_code == chess.KING:
                continue
            count_p = square_piece_deaths[sq][p_code]
            pct_p = (count_p / d_count * 100) if d_count > 0 else 0.0
            piece_breakdown[p_name] = {
                "count": count_p,
                "percentage": round(pct_p, 2)
            }

        # Game outcome win rates when combat occurs on this square
        outcomes = square_game_outcomes[sq]
        sq_total_games = sum(outcomes.values())
        w_pct = round(outcomes["white"] / sq_total_games * 100, 2) if sq_total_games > 0 else 0.0
        b_pct = round(outcomes["black"] / sq_total_games * 100, 2) if sq_total_games > 0 else 0.0
        d_pct = round(outcomes["draw"] / sq_total_games * 100, 2) if sq_total_games > 0 else 0.0

        squares_list.append({
            "square": sq,
            "deaths": d_count,
            "p_emp": float(round(p_emp, 6)),
            "p_emp_percentage": float(round(p_emp * 100, 4)),
            "piece_breakdown": piece_breakdown,
            "outcomes": {
                "white_win_pct": w_pct,
                "black_win_pct": b_pct,
                "draw_pct": d_pct,
                "total_games_with_combat": sq_total_games
            }
        })

    # Sort to determine rankings
    squares_list.sort(key=lambda s: s["deaths"], reverse=True)
    for rank_idx, sq_data in enumerate(squares_list, 1):
        sq_data["rank"] = rank_idx

    deadliest_square_info = squares_list[0]
    deadliest_sq = deadliest_square_info["square"]
    print(f"\n#1 Deadliest Square: {deadliest_sq.upper()} with {deadliest_square_info['deaths']} deaths (p_emp = {deadliest_square_info['p_emp_percentage']}%)")

    # Deep-dive on top square brawls & openings
    top_sq_brawls = brawl_records[deadliest_sq]
    brawl_outcomes = Counter(b["brawl_winner"] for b in top_sq_brawls)
    game_outcomes_after_brawl = Counter(b["game_outcome"] for b in top_sq_brawls)

    # Correlation: If White won the brawl on deadliest_sq, did White win the match?
    white_won_brawl_and_game = sum(1 for b in top_sq_brawls if b["brawl_winner"] == "white" and b["game_outcome"] == "white")
    white_won_brawl_total = sum(1 for b in top_sq_brawls if b["brawl_winner"] == "white")
    p_white_game_given_brawl = (white_won_brawl_and_game / white_won_brawl_total) if white_won_brawl_total > 0 else 0

    black_won_brawl_and_game = sum(1 for b in top_sq_brawls if b["brawl_winner"] == "black" and b["game_outcome"] == "black")
    black_won_brawl_total = sum(1 for b in top_sq_brawls if b["brawl_winner"] == "black")
    p_black_game_given_brawl = (black_won_brawl_and_game / black_won_brawl_total) if black_won_brawl_total > 0 else 0

    # Top openings for deadliest square
    top_openings = []
    for (eco, name), count in square_openings[deadliest_sq].most_common(12):
        top_openings.append({
            "eco": eco,
            "name": name,
            "death_count": count
        })

    # Top move sets
    top_movesets = []
    for moveset, count in square_movesets[deadliest_sq].most_common(8):
        top_movesets.append({
            "moves": moveset,
            "count": count
        })

    results = {
        "metadata": {
            "total_games_analyzed": total_games,
            "total_deaths_all_squares": total_deaths,
            "deadliest_square": deadliest_sq,
            "deadliest_square_rank": 1
        },
        "all_squares": squares_list,
        "deadliest_square_deepdive": {
            "square": deadliest_sq,
            "deaths": deadliest_square_info["deaths"],
            "p_emp": deadliest_square_info["p_emp"],
            "p_emp_percentage": deadliest_square_info["p_emp_percentage"],
            "piece_breakdown": deadliest_square_info["piece_breakdown"],
            "outcomes": deadliest_square_info["outcomes"],
            "brawl_statistics": {
                "total_multi_piece_brawls": len(top_sq_brawls),
                "brawl_winner_counts": dict(brawl_outcomes),
                "game_outcome_counts_after_brawl": dict(game_outcomes_after_brawl),
                "p_game_win_given_brawl_win_white": round(p_white_game_given_brawl, 4),
                "p_game_win_given_brawl_win_black": round(p_black_game_given_brawl, 4)
            },
            "top_openings": top_openings,
            "top_movesets": top_movesets
        }
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================== PART 2 ANALYSIS RESULTS ==================")
    print(f"Top 5 Deadliest Squares:")
    for sq in squares_list[:5]:
        print(f"  #{sq['rank']} {sq['square'].upper()}: {sq['deaths']} deaths ({sq['p_emp_percentage']:.2f}%)")
    print(f"\nPiece breakdown on #{deadliest_sq.upper()}:")
    for p_name, data in deadliest_square_info["piece_breakdown"].items():
        print(f"  {p_name}: {data['count']} deaths ({data['percentage']}%)")
    print(f"\nBrawl correlation on {deadliest_sq.upper()}:")
    print(f"  P(White Wins Game | White Wins Brawl on {deadliest_sq}) = {p_white_game_given_brawl * 100:.1f}%")
    print(f"  P(Black Wins Game | Black Wins Brawl on {deadliest_sq}) = {p_black_game_given_brawl * 100:.1f}%")
    print("=============================================================\n")

if __name__ == "__main__":
    run_part2()
