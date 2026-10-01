"""Resource monitoring uses the same stat record for both reported quantities."""

import pytest

from mooneural.training.generic_process_monitor import ProcessGroupMonitor


def write_stat(root, pid, group, ticks, pages, name="worker (task)"):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["0"] * 22
    fields[0], fields[2], fields[21] = "S", str(group), str(pages)
    for index, value in zip((11, 12, 13, 14), ticks, strict=True):
        fields[index] = str(value)
    (directory / "stat").write_text(f"{pid} ({name}) " + " ".join(fields))


def test_group_cpu_rss_and_disappearing_process(tmp_path):
    write_stat(tmp_path, 101, 101, (50, 20, 10, 5), 12)
    write_stat(tmp_path, 102, 101, (3, 4, 0, 0), 4)
    write_stat(tmp_path, 103, 103, (1000, 1000, 0, 0), 99)
    (tmp_path / "104").mkdir()
    (tmp_path / "self").mkdir()
    monitor = ProcessGroupMonitor(tmp_path, ticks_per_second=100, page_size=4096)
    sample = monitor.sample(101)
    assert sample.cpu_seconds == .92
    assert sample.rss_bytes == 16 * 4096
    assert sample.members == (101, 102)
    assert sample.examined_processes == 3


def test_paired_metric_calls_share_sample(tmp_path):
    write_stat(tmp_path, 101, 101, (2, 3, 0, 0), 5)
    monitor = ProcessGroupMonitor(tmp_path, ticks_per_second=100, page_size=4096)
    assert monitor.group_cpu(101) == .05
    write_stat(tmp_path, 101, 101, (20, 30, 0, 0), 50)
    assert monitor.process_group_rss(101) == 5 * 4096
    assert monitor.polls == 1
    with pytest.raises(ValueError, match="matching combined"):
        monitor.process_group_rss(101)
    assert monitor.group_cpu(101) == .5
    assert monitor.process_group_rss(101) == 50 * 4096
    assert monitor.polls == 2


def test_malformed_stat_is_explicit_failure(tmp_path):
    write_stat(tmp_path, 101, 101, (1, 2, 0, 0), 1)
    (tmp_path / "101/stat").write_text("malformed")
    with pytest.raises((ValueError, IndexError)):
        ProcessGroupMonitor(tmp_path).sample(101)


def test_unknown_group_and_invalid_units(tmp_path):
    write_stat(tmp_path, 101, 101, (1, 2, 0, 0), 1)
    monitor = ProcessGroupMonitor(tmp_path)
    assert monitor.sample(999).members == ()
    with pytest.raises(ValueError):
        monitor.sample(True)
    with pytest.raises(ValueError):
        ProcessGroupMonitor(tmp_path, ticks_per_second=0)
