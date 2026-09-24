import argparse
import os
import sys
from pathlib import Path

from .client import DEFAULT_RATSD_HOST, DEFAULT_RATSD_PORT, run_attestation_flow


def main():
    parser = argparse.ArgumentParser(
        description="Attestation client for SEV-SNP benchmark"
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="Attestation server base URL (e.g. http://10.0.2.2:8000)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for artifacts and timings",
    )
    parser.add_argument(
        "--ratsd-host",
        default=os.environ.get("RATSD_HOST", DEFAULT_RATSD_HOST),
        help="ratsd host (env RATSD_HOST, default: %(default)s)",
    )
    parser.add_argument(
        "--ratsd-port",
        type=int,
        default=int(os.environ.get("RATSD_PORT", DEFAULT_RATSD_PORT)),
        help="ratsd port (env RATSD_PORT, default: %(default)s)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    ratsd_url = f"http://{args.ratsd_host}:{args.ratsd_port}"

    try:
        print(f"Starting attestation flow with server: {args.server_url}")
        print(f"Output directory: {output_dir}")

        timings = run_attestation_flow(args.server_url, output_dir, ratsd_url)

        print("\n=== Attestation Flow Completed ===")
        print(f"Total time: {timings['total_client_time']:.3f}s")
        print("\nTiming breakdown:")
        for key, value in timings.items():
            if key != "total_client_time":
                print(f"  {key}: {value:.3f}s")
        print(f"\nArtifacts saved to: {output_dir}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
