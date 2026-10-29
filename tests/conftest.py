"""Suite-wide test environment."""

import os

# dns_race (on by default) would race real DNS servers from inside tests that
# fake getaddrinfo (the rebinding guards): keep resolution deterministic. The
# dns_race tests opt in with Settings(dns_race=True).
os.environ.setdefault("SEARCHIO_DNS_RACE", "false")
# dead_ip_ttl_s (on by default) is process-wide: a test that fakes a connect
# timeout would mark an address dead for every later test. The cache's own
# tests set it explicitly.
os.environ.setdefault("SEARCHIO_DEAD_IP_TTL_S", "0")
