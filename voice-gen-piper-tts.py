#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import os
import sys

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
import wave
from piper import PiperVoice


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate voice packs from CSV list.",
    )
    parser.add_argument("file", type=str, help="CSV Translation file")
    parser.add_argument("langdir", type=str, help="Language subfolder")
    parser.add_argument(
        "--cuda", help="Use CUDA", action=argparse.BooleanOptionalAction
    )

    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    csv_file = args.file
    csv_rows = 0
    langdir = args.langdir
    basedir = Path(__file__).resolve().parent
    outdir = Path()

    voice = PiperVoice.load("en_US-amy-low.onnx", use_cuda=args.cuda)

    if not os.path.isfile(csv_file):
        print("Error: voice file not found")
        sys.exit(1)

    # Get number of rows in CSV file
    with open(csv_file, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        reader = ((field.strip() for field in row) for row in reader)  # Strip spaces
        csv_rows = sum(1 for row in reader)

    # Drop header row from progress count if present
    csv_rows = max(csv_rows - 1, 0)
    csv_path = Path(csv_file).resolve()
    voices_root = Path(__file__).resolve().parent / "voices"

    all_csvs = sorted(voices_root.glob("*.csv")) if voices_root.exists() else []
    if not all_csvs:
        all_csvs = sorted(csv_path.parent.glob("*.csv"))

    total_files = len(all_csvs) if all_csvs else 1
    processed_files = next(
        (idx + 1 for idx, f in enumerate(all_csvs) if f.resolve() == csv_path), 1
    )

    console = Console(force_terminal=True, no_color=False)
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.fields[status]}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
    )

    class StatusLine:
        def __init__(self) -> None:
            self.message = ""

        def update(self, message: str) -> None:
            self.message = message

        def __rich_console__(self, console, options):
            yield Text(self.message)

    status_line = StatusLine()
    layout = Group(status_line, progress)

    # Process CSV file with progress bar
    with (
        open(csv_file, "rt", encoding="utf-8") as csvfile,
        Live(layout, console=console, refresh_per_second=10, transient=False),
    ):
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        reader = ((field.strip() for field in row) for row in reader)  # Strip spaces
        task_id = progress.add_task("Synthesizing", total=csv_rows or None, status="")

        def report(msg: str) -> None:
            print(msg)
            status_line.update(msg)
            progress.refresh()

        line_count = 0
        processed_count = 0

        try:
            fail_streak = 0
            for row in reader:
                row = list(row)  # Convert the generator to a list
                line_count += 1
                if line_count == 1:
                    # absorb header row
                    continue

                try:
                    path_part = row[4] if len(row) > 4 else ""
                    outdir: Path = (
                        basedir / "SOUNDS" / langdir / path_part
                        if path_part
                        else basedir / "SOUNDS" / langdir
                    )
                    en_text = row[1] if len(row) > 1 else ""
                    text = row[2] if len(row) > 2 else ""
                    filename = row[5] if len(row) > 5 else ""

                    if not filename:
                        report(
                            f"[{line_count}/{csv_rows}] Skipping row with no filename"
                        )
                        progress.update(task_id, advance=1)
                        processed_count += 1
                        continue

                    outfile: Path = outdir / filename

                    outdir.mkdir(parents=True, exist_ok=True)

                    if not text:
                        report(
                            f"[{line_count}/{csv_rows}] Skipping as no text to translate"
                        )
                        progress.update(task_id, advance=1)
                        processed_count += 1
                        continue

                    if not outfile.exists():
                        report(
                            f'[{line_count}/{csv_rows}] Translate "{en_text}" to "{text}", save as "{outfile}".'
                        )

                        with wave.open(str(outfile), "wb") as wav_file:
                            voice.synthesize_wav(text, wav_file)

                    else:
                        report(
                            f'[{line_count}/{csv_rows}] Skipping "{filename}" as already exists.'
                        )
                    progress.update(task_id, advance=1)
                    processed_count += 1
                    fail_streak = 0

                except Exception as e:
                    report(f"[{line_count}/{csv_rows}] Error processing row: {e}")
                    progress.update(task_id, advance=1)
                    processed_count += 1
                    fail_streak += 1
                    if fail_streak >= 3:
                        report("Aborting after 3 consecutive failures")
                        raise SystemExit(1)
                    continue
        except KeyboardInterrupt:
            report(
                f"Interrupted. Processed {processed_files}/{total_files} files; {processed_count}/{csv_rows} entries in current file."
            )
            progress.update(task_id, completed=processed_count)
            raise SystemExit(1)

        report(
            f'Finished processing {processed_files}/{total_files} files ({processed_count}/{csv_rows} entries) from "{csv_file}" using {os.path.basename(__file__)}.'
        )


if __name__ == "__main__":
    main()
