import argparse
import os
import sys
from pathlib import Path

import uvicorn

from .server import (
    DEFAULT_VERIFIER_HOST,
    DEFAULT_VERIFIER_PORT,
    app,
    configure,
    generate_config,
    parse_size,
)


def main():
    parser = argparse.ArgumentParser(
        description="Attestation server for SEV-SNP benchmark"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: %(default)s)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: %(default)s)",
    )
    parser.add_argument(
        "--config-size",
        default="1M",
        help="Size of the config to provision, K/M/G suffix (default: %(default)s)",
    )
    parser.add_argument("--output", help="Output directory for artifacts and timings")
    parser.add_argument(
        "--verifier-host",
        default=os.environ.get("VERIFIER_HOST", DEFAULT_VERIFIER_HOST),
        help="Veraison host (env VERIFIER_HOST, default: %(default)s)",
    )
    parser.add_argument(
        "--verifier-port",
        type=int,
        default=int(os.environ.get("VERIFIER_PORT", DEFAULT_VERIFIER_PORT)),
        help="Veraison port (env VERIFIER_PORT, default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        size_bytes = parse_size(args.config_size)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Generating {size_bytes} bytes ({args.config_size}) of configuration data..."
    )
    configure(
        data=generate_config(size_bytes),
        out_dir=Path(args.output) if args.output else None,
        verifier=f"https://{args.verifier_host}:{args.verifier_port}",
    )
    if args.output:
        print(f"Output directory: {args.output}")

    print(f"Starting attestation server on {args.host}:{args.port}")
    print(f"  - Nonce endpoint: POST http://{args.host}:{args.port}/nonce")
    print(f"  - Attest endpoint: POST http://{args.host}:{args.port}/attest")
    print(f"  - Health endpoint: GET http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
