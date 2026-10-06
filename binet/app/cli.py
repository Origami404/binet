"""The binet command line interface."""
import argparse
from datetime import datetime
import logging
from pathlib import Path

from binet import logger


def _live_command(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        args.command_parser.error("provide -- python SCRIPT [ARGS...]")
    return command


def _ls(args):
    from binet.app.ls import ls

    return ls(_live_command(args))


def _get(args):
    from binet.app.get import get

    return get(_live_command(args), args.output, args.kernel_name, num_threads=args.num_threads)


def _mv(args):
    from binet.cuda.inject import RING_DEPTH
    from binet.app.mv import mv as build

    destination = build(args.cubin, args.sites, args.output, kernel=args.kernel_name,
                        ring_depth=RING_DEPTH if args.ring_depth is None else args.ring_depth)
    print(destination)
    return 0


def _profile(args):
    command = _live_command(args)
    output = (args.output or Path.home() / ".binet" / datetime.now().strftime(args.action + "-%Y%m%d-%H%M%S-%f")).resolve()
    from binet.app.profile import profile as capture

    result = capture(args.cubin, command, output)
    for trace in result["traces"]:
        print(f"Profile: {output / trace['file']}")
    if not result["traces"]:
        logger.warning("No matching launches were profiled.")
    for warning in result["warnings"]:
        logger.warning("%s", warning)
    for error in result["errors"]:
        logger.error("%s", error)
    return result["exit_code"]


def _parser():
    parser = argparse.ArgumentParser(prog="binet", description="Inspect and instrument CUDA kernel binaries.")
    commands = parser.add_subparsers(dest="action", required=True)
    ls = commands.add_parser("ls", help="list executed CUDA kernel names and grid/block dimensions")
    ls.set_defaults(handler=_ls, command_parser=ls)
    get = commands.add_parser("get", help="dump a loaded cubin containing a kernel and write its instruction annotations")
    get.set_defaults(handler=_get, command_parser=get)
    mv = commands.add_parser("mv", help="inject probes before selected original instruction sites")
    mv.set_defaults(handler=_mv)
    profile = commands.add_parser("profile", help="profile the first matching launch with a prepared kernel patch")
    profile.set_defaults(handler=_profile, command_parser=profile)
    for command_parser in (ls, get, profile):
        command_parser.add_argument("command", nargs=argparse.REMAINDER, help="Python command to run, preceded by --")
    get.add_argument("--kernel-name", required=True, help="exact kernel symbol or exact demangled name")
    get.add_argument("--output", type=Path, required=True, help="new cubin path; annotations use the same stem with .info.json")
    get.add_argument("--num-threads", type=int, default=1024,
                     help="threads per block (1-1024); mask out unlaunched warps (default: 1024)")
    profile.add_argument("--output", type=Path, help="new output directory (default: ~/.binet/COMMAND-TIMESTAMP)")
    profile.add_argument("--cubin", type=Path, required=True, help="prepared .cubin with adjacent .meta.json")
    mv.add_argument("cubin", type=Path, help="source cubin")
    mv.add_argument("--kernel-name", help="exact kernel symbol or unique demangled substring")
    mv.add_argument("--sites", type=int, nargs="+", required=True,
                    help="positive original instruction indices to probe, e.g. --sites 12 48 77")
    mv.add_argument("--output", type=Path, required=True, help="new injected .cubin and adjacent .meta.json")
    mv.add_argument("--ring-depth", type=int, help="records per warp, a positive power of two (default: 16)")
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
