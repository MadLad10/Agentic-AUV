"""Tiny freeze recorder: one fsync'd line every few seconds, so the last state before a hard freeze survives
a forced power-off. Records available memory, swap, pressure stall info (PSI), CPU temperature and the top
processes by memory.

    setsid nohup python3 scripts/sysmon.py > /dev/null 2>&1 < /dev/null &
    tail logs/sysmon.log
"""
import glob
import os
import time

OUT = os.path.join(os.path.dirname(__file__), '..', 'logs', 'sysmon.log')
PERIOD_S = 5.0
MAX_BYTES = 20_000_000


def meminfo():
    m = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, v = line.split(':')
            m[k] = int(v.split()[0]) // 1024
    return m


def psi(name):
    with open(f'/proc/pressure/{name}') as f:
        some = f.readline().split()
    return some[1].split('=')[1]          # 'some' avg10, in percent


def max_temp():
    best = 0
    for p in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
        try:
            with open(p) as f:
                best = max(best, int(f.read()) // 1000)
        except OSError:
            pass
    return best


def top_rss(n=3):
    procs = []
    for d in glob.glob('/proc/[0-9]*'):
        try:
            with open(d + '/statm') as f:
                rss = int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') // 2**20
            with open(d + '/comm') as f:
                procs.append((rss, f.read().strip()))
        except (OSError, IndexError, ValueError):
            pass
    return ' '.join(f'{c}:{r}' for r, c in sorted(procs, reverse=True)[:n])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and os.path.getsize(OUT) > MAX_BYTES:
        os.replace(OUT, OUT + '.1')
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.write(fd, f'# boot {time.strftime("%F %T")}: time avail_MB swap_used_MB psi_mem psi_cpu psi_io load1 temp_C top\n'.encode())
    while True:
        m = meminfo()
        line = (f'{time.strftime("%T")} {m["MemAvailable"]} {m["SwapTotal"] - m["SwapFree"]} '
                f'{psi("memory")} {psi("cpu")} {psi("io")} {os.getloadavg()[0]:.1f} {max_temp()} {top_rss()}\n')
        os.write(fd, line.encode())
        os.fsync(fd)
        time.sleep(PERIOD_S)


if __name__ == '__main__':
    main()
