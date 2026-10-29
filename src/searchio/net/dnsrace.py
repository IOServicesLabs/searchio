"""dns_race: the system resolver, raced against the system's own DNS servers.

On some machines ``getaddrinfo`` for a new name stalls ~10 s while a direct
query to the very server the OS is configured with answers in milliseconds
(Windows here: servers {9.9.9.9, 1.1.1.1}, 1.1.1.1 filtered by the network,
every first contact with a host paid the resolver's fallback timeout). The
race sends one UDP A query per configured server alongside ``getaddrinfo``
and takes the first answer that carries addresses.

Never decided by the race: an NXDOMAIN, an empty or malformed reply, or no
reply at all leaves the answer to ``getaddrinfo``. Names in the hosts file,
IP literals and single-label names go to the OS only. The OS lookup keeps
running in the background, so its cache warms either way. Callers vet the
returned addresses exactly as they vet ``getaddrinfo``'s.
"""

from __future__ import annotations

import ipaddress
import os
import random
import select
import socket
import struct
import sys
import threading
import time

_SERVERS: list[str] | None = None
_HOSTS: set[str] | None = None
_LOCK = threading.Lock()
#: The servers' port (tests point it at a local stub).
_PORT = 53
#: time.monotonic() of the last usable UDP answer: a configured remote
#: server replied, so the link is up (dead_ip_ttl_s reads this).
LAST_UDP_OK = [0.0]
#: dns_nx_fast: an NXDOMAIN from one of the machine's OWN configured servers
#: ends the lookup at once -- it is the answer getaddrinfo will reach, after
#: ~10 s on this Windows resolver (every typo'd or dead domain cost 10 s).
NX_FAST = [False]
#: dns_servfail_fast: SERVFAIL from EVERY one of the machine's own servers
#: ends the lookup (EAI_AGAIN, what getaddrinfo reaches seconds later).
SERVFAIL_FAST = [False]


def rcode_of(data: bytes, tid: int) -> int | None:
    """The response code of OUR reply (None: not ours / not a reply)."""
    if len(data) < 12:
        return None
    rid, flags = struct.unpack(">HH", data[:4])
    if rid != tid or not flags & 0x8000:
        return None
    return flags & 0x000F


def _windows_servers() -> list[str]:
    """DNS servers of the interfaces that hold an address now -- a VPN or
    adapter that is down keeps its servers in the registry, and the race
    must not send names to resolvers the machine is not using. A static
    NameServer overrides the DHCP-assigned one, as Windows applies them."""
    import winreg

    def val(k, name: str) -> str:
        try:
            return str(winreg.QueryValueEx(k, name)[0] or "")
        except OSError:
            return ""

    out: list[str] = []
    base = r"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as ifs:
            i = 0
            while True:
                try:
                    guid = winreg.EnumKey(ifs, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base + "\\" + guid) as k:
                        addrs = (val(k, "DhcpIPAddress") + " " + val(k, "IPAddress")).split()
                        if not any(a not in ("0.0.0.0", "") for a in addrs):
                            continue  # no address: the adapter is down
                        servers = val(k, "NameServer") or val(k, "DhcpNameServer")
                        out += servers.replace(",", " ").split()
                except OSError:
                    continue
    except OSError:
        pass
    return out


def _posix_servers() -> list[str]:
    try:
        with open("/etc/resolv.conf", encoding="utf-8") as f:
            return [ln.split()[1] for ln in f if ln.startswith("nameserver") and len(ln.split()) > 1]
    except OSError:
        return []


def system_servers() -> list[str]:
    """The machine's configured IPv4 DNS servers, deduplicated, in order."""
    global _SERVERS
    with _LOCK:
        if _SERVERS is None:
            try:
                raw = _windows_servers() if sys.platform == "win32" else _posix_servers()
            except Exception:  # noqa: BLE001 -- no list = no race, never a failure
                raw = []
            seen: list[str] = []
            for s in raw:
                try:
                    if ipaddress.ip_address(s).version == 4 and s not in seen:
                        seen.append(s)
                except ValueError:
                    continue
            _SERVERS = seen
        return list(_SERVERS)


def _hosts_names() -> set[str]:
    global _HOSTS
    with _LOCK:
        if _HOSTS is None:
            path = (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                 r"System32\drivers\etc\hosts")
                    if sys.platform == "win32" else "/etc/hosts")
            names: set[str] = set()
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    for ln in f:
                        ln = ln.split("#", 1)[0].split()
                        names.update(n.lower() for n in ln[1:])
            except OSError:
                pass
            _HOSTS = names
        return _HOSTS


def _query(host: str) -> tuple[int, bytes]:
    tid = random.randrange(65536)
    qname = b"".join(bytes([len(p)]) + p.encode("idna") for p in host.rstrip(".").split("."))
    return tid, struct.pack(">HHHHHH", tid, 0x0100, 1, 0, 0, 0) + qname + b"\x00" + struct.pack(">HH", 1, 1)


def _skip_name(data: bytes, i: int) -> int:
    while True:
        n = data[i]
        if n == 0:
            return i + 1
        if n & 0xC0 == 0xC0:
            return i + 2
        i += n + 1


def parse_a(data: bytes, tid: int) -> list[str]:
    """IPv4 addresses in a DNS response with id ``tid`` (empty: none/unusable)."""
    if len(data) < 12:
        return []
    rid, flags, qd, an = struct.unpack(">HHHH", data[:8])
    if rid != tid or not flags & 0x8000 or flags & 0x000F:  # not ours / not a reply / rcode != 0
        return []
    i = 12
    for _ in range(qd):
        i = _skip_name(data, i) + 4
    ips = []
    for _ in range(an):
        i = _skip_name(data, i)
        rtype, _cls, _ttl, rdlen = struct.unpack(">HHIH", data[i:i + 10])
        i += 10
        if rtype == 1 and rdlen == 4:
            ips.append(socket.inet_ntoa(data[i:i + 4]))
        i += rdlen
    return ips


def _udp_race(host: str, servers: list[str], deadline: float, done: threading.Event,
              trusted_nx: frozenset[str] = frozenset(),
              trusted_servfail: frozenset[str] = frozenset()) -> list[str] | None | bool:
    """IPv4 answers; [] for no usable answer; None for an NXDOMAIN from a
    server in ``trusted_nx`` (the name does not exist); False once EVERY
    server in ``trusted_servfail`` answered SERVFAIL (the zone is broken)."""
    tid, pkt = _query(host)
    socks = []
    origin: dict = {}
    servfail: set[str] = set()
    try:
        for srv in servers:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setblocking(False)
            try:
                s.sendto(pkt, (srv, _PORT))
                socks.append(s)
                origin[s] = srv
            except OSError:
                s.close()
        live = list(socks)
        while live and not done.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                break
            ready, _, _ = select.select(live, [], [], min(left, 0.05))
            for s in ready:
                try:
                    data, _ = s.recvfrom(4096)
                except OSError:
                    live.remove(s)
                    continue
                ips = parse_a(data, tid)
                if ips:
                    LAST_UDP_OK[0] = time.monotonic()
                    return ips
                rc = rcode_of(data, tid)
                if origin.get(s) in trusted_nx and rc == 3:
                    return None
                if rc == 2 and origin.get(s) in trusted_servfail:
                    servfail.add(origin[s])
                    if servfail >= (trusted_servfail & set(origin.values())):
                        return False
                live.remove(s)  # NXDOMAIN / empty: this server says no; the OS decides
        return []
    finally:
        for s in socks:
            s.close()


def resolve(host: str, port: int, extra_servers: list[str] | None = None,
            udp_timeout_s: float = 2.5) -> list:
    """``getaddrinfo(host, port, 0, SOCK_STREAM)``-shaped answers, raced."""
    servers = system_servers() + [s for s in (extra_servers or []) if s]
    plain = (not servers or "." not in host.strip(".") or host.lower() in _hosts_names())
    if not plain:
        try:
            ipaddress.ip_address(host)
            plain = True
        except ValueError:
            pass
    if plain:
        return socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    box: dict = {}
    done = threading.Event()

    def os_lookup() -> None:
        try:
            box["os"] = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        except Exception as exc:  # noqa: BLE001 -- re-raised by the caller below
            box["err"] = exc
        done.set()

    threading.Thread(target=os_lookup, daemon=True, name="searchio-dns").start()
    sysv = frozenset(system_servers())
    ips = _udp_race(host, list(dict.fromkeys(servers)), time.monotonic() + udp_timeout_s, done,
                    sysv if NX_FAST[0] else frozenset(),
                    sysv if SERVFAIL_FAST[0] else frozenset())
    if ips is False:
        if "os" in box:
            return box["os"]
        # dns_servfail_fast: every one of the machine's own resolvers said
        # SERVFAIL (a dead or DNSSEC-broken zone: gmane.org answers SERVFAIL
        # in 16-1200 ms from both, the OS gives up only after 2-6 s with the
        # same EAI_AGAIN) -- the OS asks those same servers, so its verdict
        # is already known.
        raise socket.gaierror(socket.EAI_AGAIN, f"SERVFAIL: {host} (dns_servfail_fast)")
    if ips is None:
        if "os" in box:
            return box["os"]  # the OS answered meanwhile: its word stands
        raise socket.gaierror(socket.EAI_NONAME, f"NXDOMAIN: {host} does not exist (dns_nx_fast)")
    if ips and not done.is_set():
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
                for ip in ips]
    done.wait()
    if "os" in box:
        return box["os"]
    raise box["err"]
