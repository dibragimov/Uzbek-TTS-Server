"""python -m navoiy_server {prepare|serve|run|doctor}"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="navoiy_server")
    parser.add_argument("command", choices=["prepare", "serve", "run", "doctor"],
                        help="prepare: download + build engines for this GPU (idempotent); "
                             "serve: start replicas + API; run: prepare then serve; "
                             "doctor: show config/GPU/cache state")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = config.load(require_api_key=args.command in ("serve", "run"))
    except ValueError as exc:
        logging.getLogger("navoiy").error("configuration error: %s", exc)
        return 2

    if args.command == "doctor":
        from .server import doctor
        print(json.dumps(doctor(settings), indent=2))
        return 0
    if args.command in ("prepare", "run"):
        from .profile_build import prepare
        prepare(settings)
    if args.command in ("serve", "run"):
        from .server import serve
        serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
