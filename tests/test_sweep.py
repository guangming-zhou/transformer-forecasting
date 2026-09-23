from scripts.sweep import status_marker


def test_status_marker_is_ascii_safe_for_windows_consoles():
    assert status_marker(True) == "[OK]"
    assert status_marker(False) == "[FAIL]"
    status_marker(True).encode("ascii")
    status_marker(False).encode("ascii")
