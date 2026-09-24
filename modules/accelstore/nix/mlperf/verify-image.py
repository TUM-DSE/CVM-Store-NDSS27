"""Verify a docker image's layers against docker's own record, from the guest.

Run by mlperf.service between `docker load` and the tag the harness waits for.
Reads the layer store directly, so it also works with dockerd down. A failure
means the layers must be dropped so the next load re-extracts them.

Exit status:
    0  every layer matches
    1  at least one layer does not: the image is corrupt
    2  nothing could be checked (unsupported storage driver, no layerdb)
"""

import argparse
import gzip
import json
import os
import subprocess
import sys

DOCKER_ROOT_DEFAULT = "/var/lib/docker"

EXIT_OK = 0
EXIT_CORRUPT = 1
EXIT_UNCHECKABLE = 2

SUPPORTED_DRIVER = "overlay2"


class Uncheckable(Exception):
    """Nothing could be checked, which is not a verdict about the image."""


def diff_ids(image, docker_root, driver, docker_bin):
    """The image's layers, bottom-most first."""
    ids = _diff_ids_offline(image, docker_root, driver)
    if ids is not None:
        return ids
    out = subprocess.run(
        [docker_bin, "image", "inspect", "-f", "{{json .RootFS.Layers}}", image],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise Uncheckable(f"no such image {image}: {(out.stderr or '').strip()}")
    return json.loads(out.stdout)


def _diff_ids_offline(image, docker_root, driver):
    """`image` resolved through repositories.json and imagedb, or None."""
    repos = os.path.join(docker_root, "image", driver, "repositories.json")
    try:
        with open(repos, encoding="utf-8") as fh:
            table = json.load(fh).get("Repositories", {})
    except (OSError, json.JSONDecodeError):
        return None
    image_id = None
    for refs in table.values():
        if image in refs:
            image_id = refs[image]
            break
    if not image_id or not image_id.startswith("sha256:"):
        return None
    config = os.path.join(
        docker_root,
        "image",
        driver,
        "imagedb",
        "content",
        "sha256",
        image_id[len("sha256:") :],
    )
    try:
        with open(config, encoding="utf-8") as fh:
            return json.load(fh)["rootfs"]["diff_ids"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def layerdb_index(docker_root, driver):
    """diff_id -> [layerdb entry]; one diff_id can sit at several chainIDs."""
    index = {}
    base = os.path.join(docker_root, "image", driver, "layerdb", "sha256")
    if not os.path.isdir(base):
        raise Uncheckable(f"no layerdb at {base}")
    for chain in sorted(os.listdir(base)):
        entry = os.path.join(base, chain)
        try:
            with open(os.path.join(entry, "diff"), encoding="utf-8") as fh:
                index.setdefault(fh.read().strip(), []).append(entry)
        except OSError:
            continue
    return index


def tar_split_sizes(entry):
    """(name, size) for every file the layer's tar carried data for, or None."""
    path = os.path.join(entry, "tar-split.json.gz")
    if not os.path.exists(path):
        return None
    sizes = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") != 1:
                    continue
                size = rec.get("size") or 0
                name = rec.get("name")
                if size and name:  # size 0 is a directory, link or empty file
                    sizes.append((name, size))
    except (OSError, json.JSONDecodeError, EOFError):
        return None
    return sizes


def check_layer(entry, docker_root, driver):
    """(description, [(name, on_disk, expected)]) for one layer."""
    try:
        with open(os.path.join(entry, "cache-id"), encoding="utf-8") as fh:
            cache_id = fh.read().strip()
    except OSError as exc:
        raise Uncheckable(f"{entry}: no cache-id ({exc})") from exc

    diff_dir = os.path.join(docker_root, driver, cache_id, "diff")
    if not os.path.isdir(diff_dir):
        raise Uncheckable(f"{cache_id[:12]}: no diff directory at {diff_dir}")

    expected = tar_split_sizes(entry)
    if expected is None:
        return _check_layer_by_size(entry, diff_dir, cache_id)

    bad = []
    for name, size in expected:
        try:
            actual = os.lstat(os.path.join(diff_dir, name)).st_size
        except OSError:
            bad.append((name, "missing", size))
            continue
        if actual != size:
            bad.append((name, actual, size))
    return f"{cache_id[:12]}: {len(expected)} files checked", bad


def _check_layer_by_size(entry, diff_dir, cache_id):
    """Fallback for a layer with no tar-split: recorded size against disk."""
    try:
        with open(os.path.join(entry, "size"), encoding="utf-8") as fh:
            recorded = int(fh.read().strip())
    except (OSError, ValueError):
        raise Uncheckable(f"{cache_id[:12]}: neither tar-split nor a recorded size")

    actual = 0
    for root, _dirs, files in os.walk(diff_dir):
        for name in files:
            try:
                actual += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    # the recorded size is tar accounting, so only a gross loss is called
    if recorded and actual < recorded * 0.95:
        return (
            f"{cache_id[:12]}: no tar-split; {actual} bytes on disk of {recorded}",
            [("<whole layer>", actual, recorded)],
        )
    return f"{cache_id[:12]}: no tar-split; {actual} bytes, within 5% of {recorded}", []


def report(desc, bad):
    """Print one layer's verdict."""
    if not bad:
        print(f"mlperf-verify: ok {desc}")
        return
    print(f"mlperf-verify: CORRUPT {desc}", file=sys.stderr)
    print(f"    {len(bad)} file(s) do not match the layer's tar", file=sys.stderr)
    for name, actual, want in bad[:8]:
        print(f"    {name}: on-disk={actual} tar={want}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Verify a docker image's layers against docker's own record."
    )
    parser.add_argument("image", help="image reference, e.g. mlperf:nix")
    parser.add_argument("--docker-root", default=DOCKER_ROOT_DEFAULT)
    parser.add_argument("--driver", default=SUPPORTED_DRIVER)
    parser.add_argument("--docker", default="docker", help="docker binary")
    args = parser.parse_args()

    if args.driver != SUPPORTED_DRIVER:
        print(
            f"mlperf-verify: storage driver {args.driver!r} is not {SUPPORTED_DRIVER};"
            " nothing checked",
            file=sys.stderr,
        )
        return EXIT_UNCHECKABLE

    try:
        layers = diff_ids(args.image, args.docker_root, args.driver, args.docker)
        index = layerdb_index(args.docker_root, args.driver)

        corrupt = 0
        for diff_id in layers:
            entries = index.get(diff_id)
            if not entries:
                raise Uncheckable(f"{diff_id[:19]}: no layerdb entry")
            for entry in entries:
                desc, bad = check_layer(entry, args.docker_root, args.driver)
                corrupt += len(bad)
                report(desc, bad)
    except Uncheckable as exc:
        print(f"mlperf-verify: cannot verify {args.image}: {exc}", file=sys.stderr)
        return EXIT_UNCHECKABLE

    if corrupt:
        print(
            f"mlperf-verify: {args.image} is corrupt ({corrupt} files across"
            f" {len(layers)} layers)",
            file=sys.stderr,
        )
        return EXIT_CORRUPT
    print(f"mlperf-verify: {args.image} ok ({len(layers)} layers)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
