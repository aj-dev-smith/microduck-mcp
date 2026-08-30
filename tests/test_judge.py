"""judge.py's pure parts: paths, scripts, rubric — no box, no Gemini."""

import pytest

from microduck_mcp import judge


def test_step_of_reads_mjlab_names():
    assert judge.step_of("a/b/videos/train/rl-video-step-20000.mp4") == 20000
    assert judge.step_of("rl-video-step-0.mp4") == 0
    assert judge.step_of("the_duck_speaks.mp4") is None


def test_find_clips_script_lists_newest_first():
    s = judge.find_clips_script()
    assert "ls -t" in s and s.rstrip().endswith("head -n 5")
    assert "grep" not in s


def test_find_clips_script_match_is_literal_not_pattern():
    s = judge.find_clips_script(match="micro$duck 'stand'")
    assert "grep -F" in s
    # shell-quoted: the single quotes in the match must not terminate the arg
    assert "micro$duck" in s and s.count("grep") == 1


def test_rubric_carries_task_and_cheats():
    text = judge.RUBRIC.format(task="stand up", cheats="- freeze")
    assert "stand up" in text and "- freeze" in text
    assert "POPULATION" in text  # the multi-duck lesson, load-bearing


def test_review_refuses_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    clip = tmp_path / "rl-video-step-0.mp4"
    clip.write_bytes(b"not a real mp4")
    with pytest.raises(judge.JudgeError, match="GEMINI_API_KEY"):
        judge.review(str(clip), task="stand up")
