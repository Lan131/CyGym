# timing_utils.py
import time
import atexit
from collections import defaultdict
from contextlib import contextmanager

# Try to use psutil if available (preferred), otherwise fall back to resource.
# psutil gives RSS in bytes; resource.ru_maxrss is platform-dependent (kilobytes on Linux).
try:
    import psutil  # type: ignore

    def _get_mem_mb() -> float:
        p = psutil.Process()
        return float(p.memory_info().rss) / (1024.0 * 1024.0)
except Exception:
    try:
        import resource  # type: ignore

        def _get_mem_mb() -> float:
            # On Linux, ru_maxrss is in kilobytes. Convert to MB.
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(usage) / 1024.0
    except Exception:
        # Last-resort fallback (very small overhead)
        def _get_mem_mb() -> float:
            return 0.0

HOTSPOTS = defaultdict(float)

@contextmanager
def timing(label: str, enabled: bool = True, record: bool = True, sink=print):
    """
    Usage:
      with timing("step A"): ...
      with timing("expensive", enabled=FLAG): ...
      with timing("quiet", sink=None): ...
    Prints timing with adaptive units (s, ms, µs, ns) and current memory usage (MB).
    """
    if not enabled:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        dt_ns = time.perf_counter_ns() - t0
        if record:
            HOTSPOTS[label] += dt_ns / 1e9  # store seconds
        if sink:
            mem_mb = _get_mem_mb()
            mem_str = f" mem:{mem_mb:.2f}MB"
            if dt_ns >= 1_000_000_000:  # ≥1s
                sink(f"[t] {label}: {dt_ns/1e9:.3f}s{mem_str}")
            elif dt_ns >= 1_000_000:    # ≥1ms
                sink(f"[t] {label}: {dt_ns/1e6:.3f}ms{mem_str}")
            elif dt_ns >= 1_000:        # ≥1µs
                sink(f"[t] {label}: {dt_ns/1e3:.3f}µs{mem_str}")
            else:                       # <1µs
                sink(f"[t] {label}: {dt_ns}ns{mem_str}")

def report_hotspots(top_k: int = 30, sink=print, header: str = "=== HOTSPOTS (cum sec) ==="):
    if not sink:
        return
    sink(header)
    for label, total in sorted(HOTSPOTS.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        sink(f"{total:9.3f}s  {label}")
    # also report current memory usage at time of reporting
    try:
        mem_mb = _get_mem_mb()
        sink(f"Current memory usage: {mem_mb:.2f} MB")
    except Exception:
        pass

@atexit.register
def _print_hotspots_at_exit():
    # Prints once at program end (you can remove if undesired)
    if HOTSPOTS:
        report_hotspots()
