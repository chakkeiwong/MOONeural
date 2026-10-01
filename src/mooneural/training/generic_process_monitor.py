"""Sample process-group CPU and resident memory with one process-table read."""

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessGroupSample:
    group: int
    cpu_seconds: float
    rss_bytes: int
    members: tuple[int, ...]
    examined_processes: int
    sampling_cpu_seconds: float
    sampling_wall_seconds: float


class ProcessGroupMonitor:
    def __init__(self, proc_root="/proc", *, ticks_per_second=None, page_size=None):
        self.proc_root = Path(proc_root)
        self.ticks_per_second = os.sysconf("SC_CLK_TCK") if ticks_per_second is None else ticks_per_second
        self.page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
        if self.ticks_per_second <= 0 or self.page_size <= 0:
            raise ValueError("positive clock and page units required")
        self.latest = None
        self.polls = 0
        self.sampling_cpu_seconds = 0.
        self.maximum_sampling_wall_seconds = 0.
        self.maximum_examined_processes = 0
        self._pending_rss_group = None

    def sample(self, group):
        if type(group) is not int or group <= 0:
            raise ValueError("positive integer process group required")
        started_cpu, started_wall = time.process_time(), time.monotonic()
        ticks, resident_pages, examined = 0, 0, 0
        members = []
        with os.scandir(self.proc_root) as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                try:
                    with open(os.path.join(entry.path, "stat"), encoding="utf-8") as stream:
                        fields = stream.read().rsplit(")", 1)[1].split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                examined += 1
                if int(fields[2]) != group:
                    continue
                cpu_ticks = sum(int(fields[index]) for index in (11, 12, 13, 14))
                rss_pages = int(fields[21])
                if cpu_ticks < 0 or rss_pages < 0:
                    raise ValueError("negative process resource counters")
                ticks += cpu_ticks
                resident_pages += rss_pages
                members.append(int(entry.name))
        result = ProcessGroupSample(group, ticks / self.ticks_per_second, resident_pages * self.page_size,
            tuple(sorted(members)), examined, time.process_time() - started_cpu, time.monotonic() - started_wall)
        self.latest = result
        self.polls += 1
        self.sampling_cpu_seconds += result.sampling_cpu_seconds
        self.maximum_sampling_wall_seconds = max(self.maximum_sampling_wall_seconds, result.sampling_wall_seconds)
        self.maximum_examined_processes = max(self.maximum_examined_processes, examined)
        return result

    def group_cpu(self, group):
        result = self.sample(group)
        self._pending_rss_group = group
        return result.cpu_seconds

    def process_group_rss(self, group):
        if self._pending_rss_group != group or self.latest is None:
            raise ValueError("RSS must consume the matching combined CPU sample")
        self._pending_rss_group = None
        return self.latest.rss_bytes

    def diagnostics(self):
        return {"polls": self.polls, "sampling_cpu_seconds": self.sampling_cpu_seconds,
            "maximum_sampling_wall_seconds": self.maximum_sampling_wall_seconds,
            "maximum_examined_processes": self.maximum_examined_processes,
            "group_coverage": "all visible numeric PIDs, exact process-group match",
            "cpu_convention": "utime + stime + cutime + cstime; final wait4 remains supervisor-owned"}
