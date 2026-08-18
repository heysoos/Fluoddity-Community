"""Shared CLI flags, so both Fluoddity apps take identical bridge options."""
import argparse


def add_bridge_args(parser: argparse.ArgumentParser,
                    default_out: str = "fluoddity") -> argparse.ArgumentParser:
    group = parser.add_argument_group("vvvv bridge")
    group.add_argument("--spout-out", type=str, default=None, metavar="NAME",
                       help=f"publish frames as this Spout sender (e.g. {default_out})")
    group.add_argument("--spout-in", type=str, default=None, metavar="NAME",
                       help="receive this Spout sender as an external field source")
    group.add_argument("--no-invert", action="store_true",
                       help="do not flip vertically on send/receive "
                            "(GL is bottom-up, DX11 top-down; flip is the default)")

    group.add_argument("--osc-port", type=int, default=None, metavar="PORT",
                       help="listen for OSC parameter control on this port (e.g. 5011)")
    group.add_argument("--osc-host", type=str, default="127.0.0.1", metavar="HOST",
                       help="interface to bind OSC on (0.0.0.0 to accept remote control)")
    group.add_argument("--osc-return-port", type=int, default=5012, metavar="PORT",
                       help="port to send telemetry back to vvvv on (0 to disable)")
    group.add_argument("--osc-return-host", type=str, default="127.0.0.1", metavar="HOST")
    group.add_argument("--osc-prefix", type=str, default="/fluoddity", metavar="PREFIX")
    group.add_argument("--osc-verbose", action="store_true",
                       help="log every accepted OSC message")

    group.add_argument("--width", type=int, default=None, metavar="W")
    group.add_argument("--height", type=int, default=None, metavar="H")
    group.add_argument("--offscreen", action="store_true",
                       help="hide the window and disable vsync; drive it over OSC")
    group.add_argument("--no-vsync", action="store_true",
                       help="disable vsync. Note: physics advances per frame, so this "
                            "changes the look as well as the framerate")
    return parser


def resolve_return(args) -> tuple[str | None, int | None]:
    """Return channel as (host, port), or (None, None) when disabled."""
    if not args.osc_return_port:
        return None, None
    return args.osc_return_host, args.osc_return_port
