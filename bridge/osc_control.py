"""OSC parameter control for a GL app whose context lives on the main thread.

The OSC server runs on its own thread and only ever writes into a lock-guarded
staging dict. The GL thread calls :meth:`apply` (or :meth:`drain`) once per
frame to pick values up. Nothing here touches ModernGL.

Address scheme, for a parameter named ``SENSOR_GAIN`` under the default prefix:

    /fluoddity/sensor_gain      <float>   absolute value
    /fluoddity/n/sensor_gain    <float>   normalized 0..1, mapped through lo/hi
    /fluoddity/params           "a,b,c"   all params, absolute, spec order
    /fluoddity/n/params         "a,b,c"   all params, normalized, spec order

The normalized family is the one that matters in practice: MIDI controllers
send 0-127, vvvv normalizes to 0..1, and the ranges come from the app's own
parameter registry so the OSC feel matches the slider feel.

The bulk ``params`` addresses exist because vvvv's ``OSCsend (Devices)`` module
may not spread its Address pin. They mirror the convention already used by the
VJ setup's gfx/Neuro bridge: one address, a comma-separated string payload.

Extra parameter groups get their own bulk addresses rather than being appended
to ``params``::

    /fluoddity/<group>          "a,b,c"   absolute, group order
    /fluoddity/n/<group>        "a,b,c"   normalized, group order

Deliberately separate: appending would mean that adding a physics parameter
later silently shifts every downstream slice index in the vvvv patch. Each
group's per-parameter addresses are registered too, so nothing is bulk-only.

Free-form messages the app polls rather than binds to a parameter -- the beat
clock and the modulation matrix -- are staged by :meth:`raw` under their address
suffix. :mod:`mod_matrix` is the only consumer.
"""
import threading
from collections.abc import MutableMapping
from dataclasses import dataclass

from pythonosc import dispatcher as _dispatcher
from pythonosc import osc_server, udp_client


@dataclass(frozen=True)
class ParamSpec:
    """One externally controllable parameter.

    Args:
        name: attribute (or dict key) on the target object. The OSC address is
            derived from ``name.lower()``.
        lo, hi: range the normalized 0..1 path maps through.
        hard_min, hard_max: absolute clamps applied to *both* paths, if set.
        power: if set, the normalized path uses ``hi * t**power`` instead of a
            linear lerp — matching Fluoddity's power-scaled sliders.
        is_int: round the result. For parameters the shader reads as an int
            (cohort count, reset mode, boundary mode).
        is_bool: produce a real ``bool``. Required, not cosmetic: these fields
            are also bound to ``imgui.checkbox()``, which rejects an int and
            raises — so writing 0/1 here would crash the UI the next time the
            containing window or menu was opened.
        target: which object :meth:`OscControl.apply` writes this to —
            ``"sim"`` for SimState, ``"preferences"`` for PreferencesState.
            Show-level settings belong on preferences so that loading a physics
            config does not stomp them.
    """
    name: str
    lo: float = 0.0
    hi: float = 1.0
    hard_min: float | None = None
    hard_max: float | None = None
    power: float | None = None
    is_int: bool = False
    target: str = "sim"
    is_bool: bool = False

    @property
    def address_name(self) -> str:
        return self.name.lower()

    def from_normalized(self, t: float) -> float:
        t = min(max(float(t), 0.0), 1.0)
        if self.power is not None:
            return self.clamp(self.hi * (t ** self.power))
        return self.clamp(self.lo + (self.hi - self.lo) * t)

    def clamp(self, value: float) -> float:
        value = float(value)
        if self.hard_min is not None:
            value = max(value, self.hard_min)
        if self.hard_max is not None:
            value = min(value, self.hard_max)
        if self.is_bool:
            return value >= 0.5
        return int(round(value)) if self.is_int else value


def _as_float(value) -> float | None:
    """Coerce an OSC argument to float. vvvv sends strings surprisingly often."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes)):
        try:
            return float(value.decode() if isinstance(value, bytes) else value)
        except ValueError:
            return None
    return None


def _split_bulk(args) -> list[float]:
    """Parse a bulk payload: one comma/space-separated string, or loose floats."""
    if len(args) == 1 and isinstance(args[0], (str, bytes)):
        text = args[0].decode() if isinstance(args[0], bytes) else args[0]
        pieces = text.replace(",", " ").split()
    else:
        pieces = args
    out = []
    for piece in pieces:
        value = _as_float(piece)
        if value is not None:
            out.append(value)
    return out


class OscControl:
    """Threaded OSC receiver plus an optional return channel for telemetry.

    Args:
        params: the primary parameter registry, reachable under the bulk
            address ``params``.
        port: UDP port to listen on (vvvv's ``OSCsend`` Remote Port).
        prefix: OSC address prefix.
        host: interface to bind. Loopback by default; use "0.0.0.0" to accept
            control from another machine on the show network.
        return_host, return_port: where to send telemetry. None disables it.
        verbose: log every accepted message — useful while patching, noisy live.
        groups: extra named parameter groups, ``{"palette": [ParamSpec, ...]}``.
            Each gets its own bulk address so its slice indices are independent
            of ``params``.
        raw_addresses: address suffixes staged verbatim for a poller rather
            than mapped to a parameter, e.g. ``["clock/tick", "mod/depth"]``.
    """

    def __init__(self, params: list[ParamSpec], port: int = 5011,
                 prefix: str = "/fluoddity", host: str = "127.0.0.1",
                 return_host: str | None = "127.0.0.1",
                 return_port: int | None = 5012,
                 verbose: bool = False,
                 groups: dict[str, list[ParamSpec]] | None = None,
                 raw_addresses: list[str] | None = None):
        self.params = list(params)
        self.prefix = prefix.rstrip("/")
        self.verbose = verbose
        self.groups = {name: list(specs)
                       for name, specs in (groups or {}).items()}

        # "params" plus every extra group, flattened. apply() walks this.
        self.all_params = list(self.params)
        for specs in self.groups.values():
            self.all_params.extend(specs)

        self._by_name = {p.name: p for p in self.all_params}
        self._pending: dict[str, float] = {}
        self._normalized: dict[str, float] = {}
        self._raw: dict[str, list[float]] = {}
        self._raw_seq: dict[str, int] = {}
        self._lock = threading.Lock()
        self._commands: list[str] = []

        # Specs are looked up by address rather than bound at registration time.
        # python-osc hands registered extra-args to the handler as a *list*
        # (dispatcher.Handler.invoke), which silently turns a bound name into
        # ['NAME'] and a bound False into a truthy [False]. Deriving from the
        # address sidesteps that convention entirely.
        self._abs_by_address = {
            f"{self.prefix}/{p.address_name}": p for p in self.all_params}
        self._norm_by_address = {
            f"{self.prefix}/n/{p.address_name}": p for p in self.all_params}
        # Same reasoning for the bulk addresses: the group is derived from the
        # address, never bound into the handler registration.
        self._bulk_by_address = {f"{self.prefix}/params": self.params}
        self._bulk_norm_by_address = {f"{self.prefix}/n/params": self.params}
        for name, specs in self.groups.items():
            self._bulk_by_address[f"{self.prefix}/{name}"] = specs
            self._bulk_norm_by_address[f"{self.prefix}/n/{name}"] = specs

        disp = _dispatcher.Dispatcher()
        for address in self._abs_by_address:
            disp.map(address, self._on_absolute)
        for address in self._norm_by_address:
            disp.map(address, self._on_normalized)
        for address in self._bulk_by_address:
            disp.map(address, self._on_bulk_absolute)
        for address in self._bulk_norm_by_address:
            disp.map(address, self._on_bulk_normalized)
        for suffix in (raw_addresses or []):
            disp.map(f"{self.prefix}/{suffix.strip('/')}", self._on_raw)
        disp.map(f"{self.prefix}/cmd", self._on_command)
        disp.set_default_handler(self._on_unmapped)

        self._server = osc_server.ThreadingOSCUDPServer((host, port), disp)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fluobridge-osc", daemon=True)
        self._thread.start()

        self._client = None
        if return_host and return_port:
            try:
                self._client = udp_client.SimpleUDPClient(return_host, return_port)
            except Exception as exc:
                print(f"[fluobridge] OSC return channel disabled: {exc}")

        groups_note = ""
        if self.groups:
            groups_note = " + " + ", ".join(
                f"{len(s)} {n}" for n, s in self.groups.items())
        print(f"[fluobridge] OSC listening on {host}:{port} "
              f"({len(self.params)} params{groups_note} under {self.prefix}/)")

    # --- receiving (OSC thread) ---

    def _stage(self, name: str, value: float) -> None:
        with self._lock:
            self._pending[name] = value
        if self.verbose:
            print(f"[fluobridge] {name} = {value:.5g}")

    def _stage_normalized(self, name: str, t: float) -> None:
        """Remember the pre-mapping 0..1 value, for :mod:`mod_matrix`.

        Only the normalized address family records this, so modulation applies
        there and absolute-address writes bypass it entirely.
        """
        with self._lock:
            self._normalized[name] = min(max(float(t), 0.0), 1.0)

    def _on_absolute(self, address, *args):
        spec = self._abs_by_address.get(address)
        if spec is None or not args:
            return
        value = _as_float(args[0])
        if value is not None:
            self._stage(spec.name, spec.clamp(value))

    def _on_normalized(self, address, *args):
        spec = self._norm_by_address.get(address)
        if spec is None or not args:
            return
        value = _as_float(args[0])
        if value is not None:
            self._stage_normalized(spec.name, value)
            self._stage(spec.name, spec.from_normalized(value))

    def _on_bulk_absolute(self, address, *args):
        specs = self._bulk_by_address.get(address)
        if specs is None:
            return
        for spec, value in zip(specs, _split_bulk(args)):
            self._stage(spec.name, spec.clamp(value))

    def _on_bulk_normalized(self, address, *args):
        specs = self._bulk_norm_by_address.get(address)
        if specs is None:
            return
        for spec, value in zip(specs, _split_bulk(args)):
            self._stage_normalized(spec.name, value)
            self._stage(spec.name, spec.from_normalized(value))

    def _on_raw(self, address, *args):
        """Stage a free-form message under its suffix for a later poller."""
        suffix = address[len(self.prefix):].lstrip("/")
        values = _split_bulk(args)
        with self._lock:
            self._raw[suffix] = values
            self._raw_seq[suffix] = self._raw_seq.get(suffix, 0) + 1
        if self.verbose:
            print(f"[fluobridge] raw {suffix} = {values}")

    def _on_command(self, address, *args):
        for arg in args:
            if isinstance(arg, (str, bytes)):
                verb = arg.decode() if isinstance(arg, bytes) else arg
                with self._lock:
                    self._commands.append(verb)

    def _on_unmapped(self, address, *args):
        if self.verbose:
            print(f"[fluobridge] unmapped {address} {args}")

    # --- applying (GL thread) ---

    def drain(self) -> dict[str, float]:
        """Take everything staged since the last call."""
        with self._lock:
            if not self._pending:
                return {}
            pending, self._pending = self._pending, {}
        return pending

    def drain_commands(self) -> list[str]:
        with self._lock:
            if not self._commands:
                return []
            commands, self._commands = self._commands, []
        return commands

    def normalized(self) -> dict[str, float]:
        """Snapshot of the last 0..1 value seen on the normalized path.

        Read-only for callers; :mod:`mod_matrix` uses it as its base values.
        """
        with self._lock:
            return dict(self._normalized)

    def raw(self, suffix: str) -> list[float] | None:
        """Last payload received on a raw address, or None if never seen."""
        with self._lock:
            values = self._raw.get(suffix)
        return list(values) if values is not None else None

    def raw_seq(self, suffix: str) -> int:
        """How many messages have arrived on a raw address.

        For trigger-style addresses, where the *arrival* is the event and the
        payload is irrelevant. Comparing values instead would swallow a repeat
        of the same value -- and since vvvv's OSCsend only transmits on change,
        a bang wired to a constant would fire once and never again.
        """
        with self._lock:
            return self._raw_seq.get(suffix, 0)

    def apply(self, target, targets: dict[str, object] | None = None
              ) -> dict[str, float]:
        """Write staged values onto ``target``, then return what was applied.

        Accepts either an object with matching attributes (Fluoddity's
        ``SimState``) or a mapping with matching keys (Core's config dict).

        Args:
            target: the default destination, used for every spec whose
                ``target`` is unknown. Keeps the single-argument call working.
            targets: optional per-``ParamSpec.target`` destinations, e.g.
                ``{"sim": ui_state.sim, "preferences": ui_state.preferences}``.
        """
        applied = self.drain()
        if not applied:
            return applied
        for name, value in applied.items():
            spec = self._by_name.get(name)
            dest = target
            if targets is not None and spec is not None:
                dest = targets.get(spec.target, target)
            if isinstance(dest, MutableMapping):
                dest[name] = value
            else:
                setattr(dest, name, value)
        return applied

    # --- telemetry (GL thread) ---

    def send(self, address: str, value) -> None:
        """Send a value back to vvvv. Never raises; the show goes on."""
        if self._client is None:
            return
        try:
            self._client.send_message(address, value)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
