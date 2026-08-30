"""Run mjlab play with a guard around per-command GUI creation.

The StandUp task's command cfg produces a degenerate slider range that trips
`assert max >= value >= min` inside viser, killing the whole viewer. The sim
is unaffected — so skip any command term whose GUI fails to build and keep
the viewer alive.
"""

import sys

from mjlab.managers import command_manager

_orig = command_manager.CommandManager.create_gui


def _safe_create_gui(self, *args, **kwargs):
    try:
        return _orig(self, *args, **kwargs)
    except Exception as e:  # noqa: BLE001 - viewer must survive any GUI failure
        print(f"[viewer-patch] command GUI skipped ({type(e).__name__}: {e})")


command_manager.CommandManager.create_gui = _safe_create_gui

from mjlab.scripts.play import main  # noqa: E402

sys.argv[0] = "play"
main()
