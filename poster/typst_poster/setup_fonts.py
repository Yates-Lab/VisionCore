"""Install user-supplied, licensed fonts into this checkout's ignored .fonts/."""
import argparse
import hashlib
from pathlib import Path
import shutil

from fonts import HERE, checked_font_paths, font_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path,
                        help="Directory containing licensed Arial regular/bold and Times New Roman")
    parser.add_argument("--check", action="store_true", help="Verify already installed fonts")
    args = parser.parse_args()
    if args.from_dir:
        # Validate all inputs before installing any; case-insensitive filenames
        # support both Windows and Linux font directories.
        if not args.from_dir.is_dir():
            parser.error("--from-dir must be an existing font directory")
        candidates = {p.name.lower(): p for p in args.from_dir.iterdir() if p.is_file()}
        sources = []
        for name, spec in font_manifest().items():
            if spec["bundled"]:
                continue
            source = candidates.get(name.lower())
            if source is None or hashlib.sha256(source.read_bytes()).hexdigest() != spec["sha256"]:
                parser.error(f"Requires {name}, {spec['family']} {spec['version']}, "
                             f"SHA-256 {spec['sha256']}; missing or different font bytes")
            sources.append((source, HERE / ".fonts" / name))
        (HERE / ".fonts").mkdir(exist_ok=True)
        for source, target in sources:
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
    elif not args.check:
        parser.error("Specify --from-dir or --check")
    checked_font_paths()
    print("PASS: all seven pinned fonts available; proprietary fonts remain local and Git-ignored")


if __name__ == "__main__":
    main()
