"""Offline tests for the remote-training tools: what gets sent, what gets read.

Nothing here touches the GPU box. The two halves worth locking down are both
pure: the SHELL SCRIPT a verb constructs (a training run is hours of GPU time,
and the failure mode of a mis-built command is discovering it twelve hours
later), and the PARSE of a training log (fixtures below reproduce rsl_rl's and
tyro's real output byte-for-byte, ANSI bolding included).

The one behavior that is not about strings is the refusal: duck_train_start
must never displace a run that is already going. That is asserted here against
a faked box, because the alternative is asserting it against a real one.

    uv run --with pytest pytest tests/ --ignore=tests/live
"""

import unittest
from unittest import mock

from microduck_mcp import train


# --------------------------------------------------------------------------
# Fixtures — real output, kept verbatim
# --------------------------------------------------------------------------

# rsl_rl bolds the banner with \033[1m ... \033[0m and centers it in 80 cols;
# the labels are right-padded to 40. Two iteration blocks, so "last wins" is
# actually tested rather than assumed.
def _iteration_block(it, total, reward, length, eta, elapsed, fps):
    pad, width = 40, 80
    return (
        "#" * width + "\n"
        "\033[1m" + f" Learning iteration {it}/{total} ".center(width) + "\033[0m \n\n"
        + f"{'Total steps:':>{pad}} {it * 98304} \n"
        + f"{'Steps per second:':>{pad}} {fps} \n"
        + f"{'Collection time:':>{pad}} 0.812s \n"
        + f"{'Learning time:':>{pad}} 0.319s \n"
        + f"{'Mean value_function loss:':>{pad}} 0.0142\n"
        + f"{'Mean surrogate loss:':>{pad}} -0.0031\n"
        + f"{'Mean reward:':>{pad}} {reward}\n"
        + f"{'Mean episode length:':>{pad}} {length}\n"
        + f"{'Mean action std:':>{pad}} 0.88\n"
        + f"{'Episode_Reward/track_lin_vel:':>{pad}} 0.4821\n"
        + f"{'Episode_Reward/action_rate:':>{pad}} -0.0093\n"
        + "-" * width + "\n"
        + f"{'Iteration time:':>{pad}} 1.13s\n"
        + f"{'Time elapsed:':>{pad}} {elapsed}\n"
        + f"{'ETA:':>{pad}} {eta}\n\n"
    )


RUNNING_LOG = (
    "[duck-train] Mjlab-StandUp-Flat-MicroDuck started 2026-08-29T20:02:11-04:00\n"
    "[mdp] Patches 1-2 active: NaN-safe reward/advantage\n"
    "wandb: Currently logged in as: ajsmith.\n"
    "wandb: Tracking run with wandb version 0.18.3\n"
    "wandb: \U0001f680 View run standup-flat at "
    "https://wandb.ai/ajsmith/mjlab_microduck/runs/7k2f9abc\n"
    "wandb: \U0001f9ea View project at https://wandb.ai/ajsmith/mjlab_microduck\n"
    + _iteration_block(48, 1000, "12.34", "231.02", "01:12:40", "00:03:38", "84213")
    + _iteration_block(49, 1000, "13.07", "244.88", "01:11:55", "00:03:42", "85044")
)

# The real failure that killed a 12 h StandUp attempt at second one.
VIDEO_FLAG_LOG = (
    "/home/ajsmi/microduck_rl/.venv/lib/python3.12/site-packages/tyro/_parsers.py:353: "
    "UserWarning: The field `env.scene.entities.robot.collisions.0.geom-names-expr` "
    "is annotated with type `tuple[str, ...]`\n"
    "  warnings.warn(message)\n"
    "╭─ Missing argument ────────╮\n"
    "│ Missing value for argument '('--video',)'. Expected 1 values.  │\n"
    "╰─────────────╯\n"
    "[duck-train] exit rc=2 at 2026-08-29T19:34:02-04:00\n"
)

TMUX_LS = (
    "duck-train\t1756500000\t1\n"
    "duck-train-standup-flat-microduck\t1756512131\t1\n"
    "ajs-editor\t1756400000\t3\n"
)

LIST_ENVS = (
    "[mdp] Patches 1-2 active: NaN-safe reward/advantage\n"
    "+---------------------------------------------------------+\n"
    "|             Available Environments in mjlab             |\n"
    "+----+----------------------------------------------------+\n"
    "| #  | Task ID                                            |\n"
    "+----+----------------------------------------------------+\n"
    "| 1  | Mjlab-BallKick-Flat-MicroDuck                      |\n"
    "| 25 | Mjlab-StandUp-Flat-MicroDuck                       |\n"
    "| 37 | Mjlab-Velocity-Flat-MicroDuck-Rollers              |\n"
    "+----+----------------------------------------------------+\n"
)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

class Names(unittest.TestCase):
    def test_slug_keeps_every_distinguishing_word(self):
        # Only Mjlab- comes off. The families differ by one word, and a slug
        # that collapsed two of them would let a run refuse to start next to
        # its own twin.
        self.assertEqual(train.slug_for("Mjlab-StandUp-Flat-MicroDuck"),
                         "standup-flat-microduck")
        self.assertNotEqual(train.slug_for("Mjlab-StandUp-Flat-MicroDuck"),
                            train.slug_for("Mjlab-StandUp-Flat-Backlash-MicroDuck"))
        self.assertNotEqual(train.slug_for("Mjlab-Velocity-Flat-MicroDuck"),
                            train.slug_for("Mjlab-Velocity-Flat-MicroDuck-Rollers"))

    def test_session_round_trips_through_its_name(self):
        s = train.session_for("Mjlab-Roulade-Flat-MicroDuck")
        self.assertEqual(s, "duck-train-roulade-flat-microduck")
        self.assertEqual(train.slug_of_session(s), "roulade-flat-microduck")

    def test_bare_legacy_session_has_no_slug(self):
        # The pre-tool session is a training session we cannot name a task for.
        self.assertIsNone(train.slug_of_session("duck-train"))
        self.assertIn("~/train_*.log", train.log_globs("duck-train"))

    def test_log_globs_are_derived_not_remembered(self):
        self.assertEqual(train.log_globs("duck-train-standup-flat-microduck"),
                         ["~/logs/train_standup-flat-microduck_*.log"])

    def test_log_path_is_per_run(self):
        import datetime
        p = train.log_path("Mjlab-StandUp-Flat-MicroDuck",
                           datetime.datetime(2026, 8, 29, 20, 2, 11))
        self.assertEqual(p, "~/logs/train_standup-flat-microduck_20260829-200211.log")

    def test_a_task_id_that_could_reach_the_shell_is_refused(self):
        for bad in ("", "   ", "Mjlab-X; rm -rf ~", "$(whoami)", "a b", "'quoted'"):
            with self.assertRaises(train.TrainError):
                train.slug_for(bad)


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------

class RunnerScript(unittest.TestCase):
    def test_command_shape(self):
        s = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/logs/x.log",
                                num_envs=4096, video=True)
        self.assertIn("cd ~/microduck_rl", s)
        self.assertIn("~/.local/bin/uv run train Mjlab-StandUp-Flat-MicroDuck "
                      "--env.scene.num-envs 4096", s)
        self.assertIn("2>&1 | tee -a ~/logs/x.log", s)

    def test_video_takes_a_value(self):
        # The bug that killed a 12 h run: mjlab runs tyro with FlagConversionOff,
        # so a bare --video is a parse error, not a flag.
        s = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/l.log", video=True)
        self.assertIn("--video True", s)
        self.assertNotIn("--video 2>&1", s)
        self.assertNotIn("--video\n", s)

    def test_no_video_omits_the_flag_entirely(self):
        s = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/l.log", video=False)
        self.assertNotIn("--video", s)

    def test_pipefail_so_the_recorded_status_is_the_trainers(self):
        # Without it every run in history exits 0, because that is tee's code.
        s = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/l.log")
        self.assertTrue(s.startswith("set -o pipefail\n"))
        self.assertIn('exit rc=$?', s)

    def test_extra_args_are_split_and_quoted(self):
        s = train.runner_script(
            "Mjlab-StandUp-Flat-MicroDuck", "~/l.log", video=False,
            extra_args="--agent.load-checkpoint model_1500.pt --agent.resume True")
        self.assertIn("--agent.load-checkpoint model_1500.pt "
                      "--agent.resume True", s)

    def test_extra_args_cannot_smuggle_a_second_command(self):
        s = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/l.log",
                                video=False, extra_args="--seed '1; rm -rf ~'")
        self.assertNotIn("; rm -rf", s.replace("'1; rm -rf ~'", ""))
        self.assertIn("'1; rm -rf ~'", s)

    def test_iterations_cap(self):
        s = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/l.log",
                                iterations=5)
        self.assertIn("--agent.max_iterations 5", s)

    def test_zero_envs_is_refused(self):
        with self.assertRaises(train.TrainError):
            train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/l.log",
                                num_envs=0)


class StartScript(unittest.TestCase):
    def test_refuses_before_it_launches(self):
        s = train.start_script("duck-train-x", "~/logs/duck-train-x.sh", "true\n")
        pre = s.index("has-session")
        self.assertLess(pre, s.index("new-session"))
        self.assertIn("REFUSED", s)
        self.assertIn("exit 3", s)

    def test_never_kills_or_replaces_anything(self):
        s = train.start_script("duck-train-x", "~/logs/duck-train-x.sh", "true\n")
        for verb in ("kill-session", "kill-server", "respawn", "send-keys"):
            self.assertNotIn(verb, s)

    def test_runner_goes_in_a_quoted_heredoc(self):
        # Quoted delimiter: $? and ~ must reach the FILE literally and be
        # expanded when bash runs it, not when the launcher writes it.
        runner = train.runner_script("Mjlab-StandUp-Flat-MicroDuck", "~/logs/x.log")
        s = train.start_script("duck-train-x", "~/logs/duck-train-x.sh", runner)
        self.assertIn("<<'DUCK_RUNNER_EOF'", s)
        body = s.split("<<'DUCK_RUNNER_EOF'\n", 1)[1].split("\nDUCK_RUNNER_EOF")[0]
        self.assertEqual(body + "\n", runner)

    def test_tmux_runs_the_file_not_an_argument(self):
        s = train.start_script("duck-train-x", "~/logs/duck-train-x.sh", "true\n")
        self.assertIn("tmux new-session -d -s duck-train-x bash -l "
                      "~/logs/duck-train-x.sh", s)

    def test_no_set_e_so_the_refusal_keeps_its_exit_code(self):
        # Under `set -e`, an `exit 3` from inside an `if` reaches ssh as 1.
        s = train.start_script("duck-train-x", "~/logs/duck-train-x.sh", "true\n")
        self.assertNotIn("set -e", s)
        self.assertIn("|| exit", s)


class WrapScript(unittest.TestCase):
    def test_the_script_is_braced_so_stdin_is_drained_before_it_runs(self):
        # A script read from stdin is read as it runs: an early exit leaves
        # the rest unread, the writer takes an EPIPE, and ssh reports its own
        # status instead of the script's.
        w = train.wrap_script("echo hi\nexit 3\n")
        self.assertEqual(w, "{\necho hi\nexit 3\n}\n")


class StatusAndStopScripts(unittest.TestCase):
    def test_status_is_read_only(self):
        s = train.status_script("duck-train-standup-flat-microduck")
        for verb in ("kill-session", "send-keys", "new-session", "rm ", "kill "):
            self.assertNotIn(verb, s)

    def test_status_without_a_session_only_lists(self):
        s = train.status_script()
        self.assertIn("tmux ls", s)
        self.assertNotIn("tail -n", s)

    def test_status_finds_the_log_from_the_session_name(self):
        s = train.status_script("duck-train-standup-flat-microduck", tail=50)
        self.assertIn("~/logs/train_standup-flat-microduck_*.log", s)
        self.assertIn("tail -n 50", s)

    def test_stop_interrupts_before_it_kills(self):
        s = train.stop_script("duck-train-x", grace_s=5)
        self.assertLess(s.index("send-keys -t duck-train-x C-c"),
                        s.index("kill-session"))
        self.assertIn("sleep 5", s)

    def test_stop_on_a_missing_session_is_a_refusal_not_a_kill(self):
        s = train.stop_script("duck-train-x")
        self.assertLess(s.index("NO SESSION"), s.index("kill-session"))
        self.assertIn("exit 4", s)


class SshArgv(unittest.TestCase):
    def test_script_travels_on_stdin_so_nothing_re_parses_it(self):
        argv = train.ssh_argv(host="duck-4090-wsl", via_wsl=False)
        self.assertEqual(argv[-2:], ["duck-4090-wsl", "bash -l -s"])
        self.assertIn("BatchMode=yes", argv)

    def test_wsl_fallback_hops_through_powershell_without_quoting_anything(self):
        argv = train.ssh_argv(host="duck-4090", via_wsl=True)
        self.assertEqual(argv[-1], "wsl -d Ubuntu -- bash -l -s")
        # No shell metacharacter PowerShell could eat.
        self.assertNotIn("|", argv[-1])
        self.assertNotIn("'", argv[-1])


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

class ParseLog(unittest.TestCase):
    def test_reads_the_latest_iteration_not_the_first(self):
        f = train.parse_log(RUNNING_LOG)
        self.assertEqual((f["iteration"], f["total_iterations"]), (49, 1000))
        self.assertEqual(f["mean_reward"], 13.07)
        self.assertEqual(f["mean_episode_length"], 244.88)
        self.assertEqual(f["eta"], "01:11:55")
        self.assertEqual(f["elapsed"], "00:03:42")
        self.assertEqual(f["steps_per_second"], 85044.0)
        self.assertNotIn("failed", f)

    def test_wandb_url_and_the_export_run_path(self):
        f = train.parse_log(RUNNING_LOG)
        self.assertEqual(f["wandb_url"],
                         "https://wandb.ai/ajsmith/mjlab_microduck/runs/7k2f9abc")
        # The argument scripts/export.py wants — the point of extracting it.
        self.assertEqual(f["wandb_run_path"],
                         "ajsmith/mjlab_microduck/7k2f9abc")

    def test_view_project_is_not_the_run(self):
        url, _ = train.parse_wandb(
            "wandb: View project at https://wandb.ai/a/b\n"
            "wandb: View run at https://wandb.ai/a/b/runs/xyz\n")
        self.assertTrue(url.endswith("/runs/xyz"))

    def test_every_shape_wandb_has_printed_that_line_in(self):
        for line in ("wandb: View run at https://wandb.ai/a/b/runs/xyz",
                     "wandb: \U0001f680 View run rosy-sky-1 at "
                     "https://wandb.ai/a/b/runs/xyz",
                     "wandb: \U0001f680 View run rosy-sky-1 at: "
                     "https://wandb.ai/a/b/runs/xyz"):
            url, path = train.parse_wandb(line + "\n")
            self.assertEqual(url, "https://wandb.ai/a/b/runs/xyz", line)
            self.assertEqual(path, "a/b/xyz", line)

    def test_a_log_with_no_iterations_reports_none_not_zero(self):
        f = train.parse_log("wandb: starting\n")
        self.assertNotIn("iteration", f)
        self.assertNotIn("mean_reward", f)

    def test_the_video_flag_failure_is_diagnosed(self):
        f = train.parse_log(VIDEO_FLAG_LOG)
        self.assertEqual(f["exit_rc"], 2)
        self.assertIn("--video", f["failed"])
        self.assertIn("FlagConversionOff", f["failed"])

    def test_disk_and_oom_are_diagnosed(self):
        self.assertIn("disk full", train.parse_log(
            "OSError: [Errno 28] No space left on device\n")["failed"])
        self.assertIn("num-envs", train.parse_log(
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate...\n"
        )["failed"])

    def test_a_traceback_is_reported_even_without_a_known_cause(self):
        f = train.parse_log("Traceback (most recent call last):\n  File ...\n")
        self.assertIn("raised", f["failed"])

    def test_a_clean_exit_clears_a_stale_diagnosis(self):
        f = train.parse_log("Traceback (most recent call last):\n"
                            "[duck-train] exit rc=0 at 2026-08-29T23:00:00-04:00\n")
        self.assertNotIn("failed", f)
        self.assertEqual(f["exit_rc"], 0)


class ParseSessions(unittest.TestCase):
    def test_only_training_sessions_are_reported(self):
        s = train.parse_sessions(TMUX_LS)
        self.assertEqual([d["session"] for d in s],
                         ["duck-train", "duck-train-standup-flat-microduck"])
        self.assertIsNone(s[0]["slug"])
        self.assertEqual(s[1]["slug"], "standup-flat-microduck")
        self.assertTrue(s[1]["created"].startswith("20"))

    def test_no_tmux_server_is_no_sessions_not_an_error(self):
        self.assertEqual(
            train.parse_sessions("no server running on /tmp/tmux-1000/default"), [])


class ParseStatus(unittest.TestCase):
    def _blob(self, alive="ALIVE", log="~/logs/train_standup-flat-microduck_1.log",
              body=RUNNING_LOG):
        return (f"=== sessions\n{TMUX_LS}"
                f"=== session duck-train-standup-flat-microduck\n{alive}\n"
                f"=== log {log}\n{body}")

    def test_a_live_run(self):
        st = train.parse_status(self._blob(),
                                "duck-train-standup-flat-microduck")
        self.assertTrue(st["alive"])
        self.assertEqual(st["logfile"],
                         "~/logs/train_standup-flat-microduck_1.log")
        self.assertEqual(st["iteration"], 49)
        self.assertEqual(len(st["sessions"]), 2)
        self.assertIn("Learning iteration", st["tail"])

    def test_a_run_that_ended(self):
        st = train.parse_status(self._blob(alive="GONE", body=VIDEO_FLAG_LOG),
                                "duck-train-standup-flat-microduck")
        self.assertFalse(st["alive"])
        self.assertEqual(st["exit_rc"], 2)
        self.assertIn("--video", st["failed"])

    def test_no_log_yet(self):
        st = train.parse_status(self._blob(log="none", body=""),
                                "duck-train-standup-flat-microduck")
        self.assertIsNone(st["logfile"])
        self.assertIsNone(st["tail"])

    def test_listing_only(self):
        st = train.parse_status(f"=== sessions\n{TMUX_LS}")
        self.assertEqual(len(st["sessions"]), 2)
        self.assertNotIn("alive", st)


class ParseTasks(unittest.TestCase):
    def test_table_rows_only(self):
        self.assertEqual(train.parse_tasks(LIST_ENVS),
                         ["Mjlab-BallKick-Flat-MicroDuck",
                          "Mjlab-StandUp-Flat-MicroDuck",
                          "Mjlab-Velocity-Flat-MicroDuck-Rollers"])


# --------------------------------------------------------------------------
# The verbs, against a faked box
# --------------------------------------------------------------------------

class Verbs(unittest.TestCase):
    def setUp(self):
        train._TASKS_CACHE.clear()

    def test_dry_run_sends_nothing(self):
        with mock.patch.object(train, "run_script") as rs:
            out = train.start("Mjlab-StandUp-Flat-MicroDuck", dry_run=True)
        rs.assert_not_called()
        self.assertFalse(out["started"])
        self.assertIn("uv run train Mjlab-StandUp-Flat-MicroDuck", out["script"])

    def test_smoke_is_64_envs_and_5_iterations(self):
        out = train.start("Mjlab-StandUp-Flat-MicroDuck", smoke=True, dry_run=True)
        self.assertEqual(out["num_envs"], train.SMOKE_NUM_ENVS)
        self.assertIn("--env.scene.num-envs 64", out["script"])
        self.assertIn("--agent.max_iterations 5", out["script"])
        # Five iterations of video is five iterations of nothing.
        self.assertNotIn("--video", out["script"])

    def test_smoke_overrides_a_num_envs_the_caller_asked_for(self):
        out = train.start("Mjlab-StandUp-Flat-MicroDuck", num_envs=4096,
                          smoke=True, dry_run=True)
        self.assertEqual(out["num_envs"], 64)

    def test_an_existing_session_is_refused_never_replaced(self):
        with mock.patch.object(train, "run_script",
                               return_value=(3, "REFUSED session "
                                             "duck-train-standup-flat-microduck "
                                             "already exists\n")):
            with self.assertRaises(train.TrainError) as e:
                train.start("Mjlab-StandUp-Flat-MicroDuck")
        self.assertIn("already going", str(e.exception))
        self.assertIn("duck_train_stop", str(e.exception))

    def test_a_successful_launch_reports_where_to_look(self):
        with mock.patch.object(train, "run_script",
                               return_value=(0, "INIT=systemd\nSTARTED\n")):
            out = train.start("Mjlab-StandUp-Flat-MicroDuck")
        self.assertTrue(out["started"])
        self.assertEqual(out["session"], "duck-train-standup-flat-microduck")
        self.assertTrue(out["logfile"].startswith("~/logs/train_standup-flat-"))
        self.assertIsNone(out.get("warning"))
        self.assertNotIn("script", out)

    def test_a_distro_without_systemd_is_warned_about(self):
        # WSL tears the distro down seconds after the last session ends, and
        # detached tmux goes with it — a launch that "worked" is not a run.
        with mock.patch.object(train, "run_script",
                               return_value=(0, "INIT=init(Ubuntu)\nSTARTED\n")):
            out = train.start("Mjlab-StandUp-Flat-MicroDuck")
        self.assertIn("systemd", out["warning"])
        self.assertIn("will NOT survive", out["warning"])

    def test_stop_refuses_a_session_that_is_not_there(self):
        with mock.patch.object(train, "run_script",
                               return_value=(4, "NO SESSION duck-train-x\n")):
            with self.assertRaises(train.TrainError):
                train.stop("duck-train-x")

    def test_stop_says_what_survives(self):
        with mock.patch.object(train, "run_script", return_value=(0, "STOPPED\n")):
            out = train.stop("duck-train-x")
        self.assertTrue(out["stopped"])
        self.assertIn("checkpoint", out["note"])

    def test_tasks_are_cached_because_list_envs_imports_every_env(self):
        with mock.patch.object(train, "run_script",
                               return_value=(0, LIST_ENVS)) as rs:
            first = train.tasks()
            second = train.tasks()
            self.assertEqual(rs.call_count, 1)
            self.assertEqual(first["tasks"], second["tasks"])
            train.tasks(refresh=True)
            self.assertEqual(rs.call_count, 2)
        self.assertEqual(first["count"], 3)

    def test_an_empty_registry_is_an_error_not_an_empty_list(self):
        with mock.patch.object(train, "run_script", return_value=(1, "boom\n")):
            with self.assertRaises(train.TrainError):
                train.tasks()

    def test_an_unresolvable_host_says_what_to_do_about_it(self):
        class P:
            returncode, stdout, stderr = 255, "", \
                "ssh: Could not resolve hostname duck-4090-wsl: nodename nor ...\n"
        with mock.patch.object(train.subprocess, "run", return_value=P()):
            with self.assertRaises(train.TrainError) as e:
                train.run_script("true")
        self.assertIn("DUCK_TRAIN_VIA_WSL", str(e.exception))


if __name__ == "__main__":
    unittest.main()
