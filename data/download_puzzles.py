import requests
import zstandard as zstd
import pandas as pd


def download_interesting_lichess_puzzles(
    output_file: str = "interesting_lichess_puzzles.csv",
    target_rows: int = 100_000,
    chunk_size: int = 50_000,
):
    """
    Stream the Lichess puzzle database, filter for tactically
    interesting positions, and save them to a local CSV.
    """

    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

    themes = (
        "pin|fork|mate|skewer|discoveredAttack|"
        "xRayAttack|promotion|backRankMate|"
        "sacrifice|deflection|attraction"
    )

    print("Streaming puzzle database...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    dctx = zstd.ZstdDecompressor()

    filtered_chunks = []
    total_rows = 0

    with dctx.stream_reader(response.raw) as reader:

        chunks = pd.read_csv(reader, chunksize=chunk_size)

        for i, chunk in enumerate(chunks):

            filtered = chunk[
                chunk["Themes"].str.contains(themes, na=False)
            ]

            filtered_chunks.append(filtered)

            total_rows += len(filtered)

            print(
                f"Processed chunk {i + 1} | "
                f"Found {len(filtered)} matching rows | "
                f"Total saved: {total_rows}"
            )

            if total_rows >= target_rows:
                break

    df = pd.concat(filtered_chunks).head(target_rows)

    df.to_csv(output_file, index=False)

    print(f"\nSaved {len(df)} rows to {output_file}")

    return df


if __name__ == "__main__":

    download_interesting_lichess_puzzles(
        output_file="interesting_lichess_puzzles.csv",
        target_rows=100_000,
    )