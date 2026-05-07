"""
Download games from lichess as PGN. Filters for high quality games
"""

import io
import re
import sys

import requests
import zstandard as zstd

# ---- Config ---------------------------------------------------------------

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst"
OUTPUT_PATH = "filtered_games.pgn"
TARGET_GAMES = 20000 # each game under our constraints has approx 75 positions thus 50000 * 75 = 3.75M positions
MIN_RATING = 1800
MIN_BASE_SECONDS = 180  # time format for each game is at least 3 mins per person

# ---- Header parsing -------------------------------------------------------

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]$')
TIME_CONTROL_RE = re.compile(r"^(\d+)(?:\+(\d+))?$")


def parse_headers(header_lines):
    """Extract tag pairs from a list of PGN header lines."""
    headers = {}
    for line in header_lines:
        m = HEADER_RE.match(line.strip())
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


def passes_filter(headers):
    """Return True if game meets rating + time control thresholds."""
    # Rating check — both players
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if white_elo < MIN_RATING or black_elo < MIN_RATING:
        return False

    # Time control check — format is "base+increment" or "base" or "-"
    tc = headers.get("TimeControl", "")
    m = TIME_CONTROL_RE.match(tc)
    if not m:
        return False  # correspondence ("-") or malformed
    base = int(m.group(1))
    if base < MIN_BASE_SECONDS:
        return False

    return True


# ---- Streaming game iterator ---------------------------------------------

def iter_games(text_stream):
    """
    Yield (headers_block, moves_block) tuples from a streaming PGN text source.

    A PGN game = header lines (starting with '['), blank line, move lines,
    blank line. We accumulate lines and emit a game when we see the
    header->moves->blank transition.
    """
    header_lines = []
    move_lines = []
    in_moves = False

    for line in text_stream:
        stripped = line.rstrip("\n")

        if not in_moves:
            if stripped.startswith("["):
                header_lines.append(line)
            elif stripped == "" and header_lines:
                # End of headers, start of moves
                in_moves = True
            # else: stray blank before any headers — skip
        else:
            if stripped == "" and move_lines:
                # End of game
                yield "".join(header_lines), "".join(move_lines)
                header_lines = []
                move_lines = []
                in_moves = False
            else:
                move_lines.append(line)

    # Trailing game with no final newline
    if header_lines and move_lines:
        yield "".join(header_lines), "".join(move_lines)


# ---- Main -----------------------------------------------------------------

def download_games(output_path):
    print(f"Streaming from {URL}")

    kept_games = 0
    seen_games = 0

    with requests.get(URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        # Streaming zstd decompressor reading from the HTTP stream
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(resp.raw) as decompressed:
            # Wrap binary decompressed stream as text, line-by-line
            text_stream = io.TextIOWrapper(decompressed, encoding="utf-8", errors="replace")

            with open(output_path, "w", encoding="utf-8") as out:
                for headers_block, moves_block in iter_games(text_stream):
                    seen_games += 1

                    headers = parse_headers(headers_block.splitlines())
                    if passes_filter(headers):
                        game_text = headers_block + "\n" + moves_block + "\n"
                        out.write(game_text)
                        kept_games += 1

                    if seen_games % 50_000 == 0:
                        print(f"  scanned {seen_games:>10,} | kept {kept_games:>8,} ")

                    if kept_games >= TARGET_GAMES:
                        break

    print()
    print(f"Done. Wrote {kept_games:,} games "
          f"to {output_path}")
    print(f"Filter pass rate: {kept_games / seen_games:.1%} "
          f"({kept_games:,} kept of {seen_games:,} scanned)")


if __name__ == "__main__":
    try:
        download_games(OUTPUT_PATH)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)