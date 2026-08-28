import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "runner.py"
sys.path.insert(0, str(REPOSITORY))
import runner as harness_runner  # noqa: E402


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary_directory.name)
        self.manifest = self.work / "suite.json"
        self.results = self.work / "results"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_manifest(self, tests, defaults=None, default_suites=None):
        document = {
            "schema_version": 1,
            "defaults": defaults or {"timeout": 5},
            "tests": tests,
        }
        if default_suites is not None:
            document["default_suites"] = default_suites
        self.manifest.write_text(json.dumps(document), encoding="utf-8")

    def invoke(self, *arguments):
        environment = dict(os.environ)
        # Keep unit tests independent of an installed PyTorch/frameworks stack;
        # production runs use the torch.xpu fallback when no count is supplied.
        environment["FRAMEWORKS_TEST_XPUS"] = "0"
        return subprocess.run(
            [sys.executable, str(RUNNER), "--manifest", str(self.manifest), *arguments],
            cwd=str(self.work),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )

    def summaries(self):
        return sorted(self.results.glob("*/summary.json"))

    def read_only_summary(self):
        summaries = self.summaries()
        self.assertEqual(len(summaries), 1, summaries)
        return json.loads(summaries[0].read_text(encoding="utf-8"))

    @staticmethod
    def passing_command(text="ok"):
        return [sys.executable, "-c", "print({!r})".format(text)]

    def test_list_combines_suite_tag_and_id_globs(self):
        self.write_manifest(
            [
                {
                    "id": "alpha-pass",
                    "suite": "cpu-basic",
                    "tags": ["smoke", "fast"],
                    "command": self.passing_command(),
                },
                {
                    "id": "alpha-slow",
                    "suite": "cpu-basic",
                    "tags": ["slow"],
                    "command": self.passing_command(),
                },
                {
                    "id": "gpu-pass",
                    "suite": "gpu",
                    "tags": ["smoke"],
                    "command": self.passing_command(),
                },
            ]
        )

        completed = self.invoke(
            "list", "--suite", "cpu-*", "--tag", "smo*", "--id", "*-pass"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("alpha-pass", completed.stdout)
        self.assertNotIn("alpha-slow", completed.stdout)
        self.assertNotIn("gpu-pass", completed.stdout)
        self.assertIn("1 test(s) selected", completed.stdout)

    def test_default_suites_limit_run_but_list_and_all_include_every_test(self):
        tests = [
            {
                "id": "quick",
                "suite": "smoke",
                "tags": [],
                "command": self.passing_command(),
            },
            {
                "id": "expensive",
                "suite": "distributed",
                "tags": [],
                "command": self.passing_command(),
            },
        ]
        self.write_manifest(tests, default_suites=["smoke"])

        listed = self.invoke("list")
        self.assertEqual(listed.returncode, 0, listed.stdout + listed.stderr)
        self.assertIn("quick", listed.stdout)
        self.assertIn("expensive", listed.stdout)

        default_results = self.work / "default-results"
        default_run = self.invoke(
            "run", "--no-module", "--results-dir", str(default_results)
        )
        self.assertEqual(default_run.returncode, 0, default_run.stdout + default_run.stderr)
        default_summary_path = next(default_results.glob("*/summary.json"))
        default_summary = json.loads(default_summary_path.read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in default_summary["tests"]], ["quick"])
        self.assertEqual(default_summary["selection"]["source"], "manifest default_suites")

        all_results = self.work / "all-results"
        all_run = self.invoke(
            "run", "--no-module", "--all", "--results-dir", str(all_results)
        )
        self.assertEqual(all_run.returncode, 0, all_run.stdout + all_run.stderr)
        all_summary_path = next(all_results.glob("*/summary.json"))
        all_summary = json.loads(all_summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["id"] for item in all_summary["tests"]], ["quick", "expensive"]
        )
        self.assertEqual(all_summary["selection"]["source"], "explicit --all")

        conflict = self.invoke("doctor", "--no-module", "--all", "--suite", "smoke")
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("--all cannot be combined", conflict.stderr)

    def test_run_records_pass_fail_and_requirement_skip(self):
        self.write_manifest(
            [
                {
                    "id": "passes",
                    "suite": "unit",
                    "tags": ["smoke"],
                    "command": self.passing_command("pass output"),
                },
                {
                    "id": "fails",
                    "suite": "unit",
                    "tags": ["negative"],
                    "command": [
                        sys.executable,
                        "-c",
                        "import sys; print('failure output'); sys.exit(7)",
                    ],
                },
                {
                    "id": "skips",
                    "suite": "unit",
                    "tags": ["requirements"],
                    "command": self.passing_command("must not run"),
                    "required_commands": ["command-that-cannot-possibly-exist-5f745b"],
                },
            ]
        )

        completed = self.invoke(
            "run", "--no-module", "--results-dir", str(self.results)
        )

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        summary = self.read_only_summary()
        self.assertEqual(summary["totals"], {"pass": 1, "fail": 1, "skip": 1, "timeout": 0})
        by_id = {result["id"]: result for result in summary["tests"]}
        self.assertEqual(by_id["passes"]["status"], "pass")
        self.assertEqual(by_id["fails"]["status"], "fail")
        self.assertEqual(by_id["fails"]["returncode"], 7)
        self.assertEqual(by_id["skips"]["status"], "skip")
        self.assertIn("required command not found", by_id["skips"]["reason"])
        self.assertIn(
            "pass output", Path(by_id["passes"]["log"]).read_text(encoding="utf-8")
        )
        self.assertIn(
            "failure output", Path(by_id["fails"]["log"]).read_text(encoding="utf-8")
        )

    def test_manifest_errors_fail_even_when_optional_requirements_are_missing(self):
        self.write_manifest(
            [
                {
                    "id": "broken-case",
                    "suite": "unit",
                    "tags": [],
                    "command": self.passing_command(),
                    "cwd": "directory-that-does-not-exist",
                    "required_commands": ["command-that-cannot-possibly-exist-7f9bcb"],
                },
                {
                    "id": "missing-declared-command",
                    "suite": "unit",
                    "tags": [],
                    "command": ["command-that-cannot-possibly-exist-a38963"],
                    "required_commands": ["command-that-cannot-possibly-exist-a38963"],
                }
            ]
        )

        completed = self.invoke(
            "run", "--no-module", "--results-dir", str(self.results)
        )

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        results = {
            result["id"]: result for result in self.read_only_summary()["tests"]
        }
        self.assertEqual(results["broken-case"]["status"], "fail")
        self.assertIn(
            "working directory does not exist", results["broken-case"]["reason"]
        )
        self.assertEqual(results["missing-declared-command"]["status"], "skip")
        self.assertIn(
            "required command not found",
            results["missing-declared-command"]["reason"],
        )

    def test_timeout_is_reported_and_returns_failure(self):
        self.write_manifest(
            [
                {
                    "id": "times-out",
                    "suite": "unit",
                    "tags": [],
                    "command": [sys.executable, "-c", "import time; time.sleep(5)"],
                    "timeout": 0.1,
                }
            ]
        )

        started = time.monotonic()
        completed = self.invoke(
            "run", "--no-module", "--results-dir", str(self.results)
        )
        elapsed = time.monotonic() - started

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertLess(elapsed, 4)
        result = self.read_only_summary()["tests"][0]
        self.assertEqual(result["status"], "timeout")
        self.assertIn("exceeded timeout", result["reason"])

    def test_case_working_directory_and_environment_are_applied(self):
        case_directory = self.work / "case-directory"
        case_directory.mkdir()
        output = self.work / "context.json"
        script = (
            "import json, os, pathlib, sys; "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps("
            "{'cwd': os.getcwd(), 'value': os.environ.get('CASE_VALUE'), "
            "'artifact': os.environ.get('FRAMEWORKS_TEST_ARTIFACT_DIR'), "
            "'port': os.environ.get('MASTER_PORT')}))"
        )
        self.write_manifest(
            [
                {
                    "id": "context",
                    "suite": "unit",
                    "tags": [],
                    "command": [sys.executable, "-c", script, str(output)],
                    "cwd": "case-directory",
                    "env": {"CASE_VALUE": "from-manifest"},
                }
            ]
        )

        completed = self.invoke(
            "run", "--no-module", "--results-dir", str(self.results)
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        context = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(context["cwd"], str(case_directory))
        self.assertEqual(context["value"], "from-manifest")
        self.assertTrue(Path(context["artifact"]).is_dir())
        self.assertTrue(1 <= int(context["port"]) <= 65535)
        result = self.read_only_summary()["tests"][0]
        self.assertEqual(result["artifact_directory"], context["artifact"])

    def test_capacity_and_python_module_requirements(self):
        self.write_manifest(
            [
                {
                    "id": "capacity-ready",
                    "suite": "requirements",
                    "tags": [],
                    "command": self.passing_command(),
                    "required_commands": [sys.executable],
                    "required_python_modules": ["json"],
                    "min_xpus": 2,
                    "min_nodes": 2,
                },
                {
                    "id": "module-missing",
                    "suite": "requirements",
                    "tags": [],
                    "command": self.passing_command(),
                    "required_python_modules": ["module_that_does_not_exist_f1756e"],
                },
            ]
        )

        doctor = self.invoke(
            "doctor",
            "--no-module",
            "--available-xpus",
            "2",
            "--available-nodes",
            "2",
            "--id",
            "capacity-*",
        )
        self.assertEqual(doctor.returncode, 0, doctor.stdout + doctor.stderr)
        self.assertIn("[READY  ] capacity-ready", doctor.stdout)

        completed = self.invoke(
            "run",
            "--no-module",
            "--available-xpus",
            "2",
            "--available-nodes",
            "2",
            "--results-dir",
            str(self.results),
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = self.read_only_summary()
        self.assertEqual(summary["totals"]["pass"], 1)
        self.assertEqual(summary["totals"]["skip"], 1)
        missing = next(item for item in summary["tests"] if item["id"] == "module-missing")
        self.assertIn("is unavailable", missing["reason"])

    def test_capacity_falls_back_to_loaded_python_torch_probe(self):
        probe_result = subprocess.CompletedProcess(
            args=["/module/bin/python"], returncode=0, stdout="4\n", stderr=""
        )
        with mock.patch.object(
            harness_runner.shutil, "which", return_value="/module/bin/python"
        ), mock.patch.object(
            harness_runner.subprocess, "run", return_value=probe_result
        ) as run_probe:
            capacity = harness_runner.detect_capacity(
                {"PATH": "/module/bin", "MODULE_ENV": "loaded"}, None, None
            )

        self.assertEqual(capacity.xpus, 4)
        self.assertEqual(capacity.xpus_source, "torch.xpu.device_count()")
        arguments, keywords = run_probe.call_args
        self.assertEqual(arguments[0][0], "/module/bin/python")
        self.assertIn("torch.xpu.device_count", arguments[0][2])
        self.assertEqual(keywords["env"]["MODULE_ENV"], "loaded")
        self.assertEqual(keywords["timeout"], 10)

    def test_dpctl_discovery_prevents_broken_torch_from_hiding_xpus(self):
        probe_results = [
            subprocess.CompletedProcess(
                args=["/module/bin/python"], returncode=0, stdout="0\n", stderr=""
            ),
            subprocess.CompletedProcess(
                args=["/module/bin/python"], returncode=0, stdout="4\n", stderr=""
            ),
        ]
        with mock.patch.object(
            harness_runner.shutil, "which", return_value="/module/bin/python"
        ), mock.patch.object(
            harness_runner.subprocess, "run", side_effect=probe_results
        ) as run_probe:
            capacity = harness_runner.detect_capacity(
                {"PATH": "/module/bin", "MODULE_ENV": "loaded"}, None, None
            )

        self.assertEqual(capacity.xpus, 4)
        self.assertEqual(capacity.xpus_source, "dpctl Level Zero discovery")
        self.assertEqual(run_probe.call_count, 2)
        self.assertIn("torch.xpu.device_count", run_probe.call_args_list[0].args[0][2])
        self.assertIn("dpctl.get_devices", run_probe.call_args_list[1].args[0][2])

    def test_capacity_counts_unique_pbs_nodefile_hosts(self):
        nodefile = self.work / "pbs_nodes"
        nodefile.write_text("node-a\nnode-a\nnode-b\nnode-b\n", encoding="utf-8")

        capacity = harness_runner.detect_capacity(
            {"FRAMEWORKS_TEST_XPUS": "0", "PBS_NODEFILE": str(nodefile)},
            None,
            None,
        )

        self.assertEqual(capacity.nodes, 2)
        self.assertEqual(capacity.nodes_source, "PBS_NODEFILE")

    def test_all_requirement_skips_return_environment_error(self):
        missing = "command-that-cannot-possibly-exist-b194cc"
        self.write_manifest(
            [
                {
                    "id": "unavailable",
                    "suite": "unit",
                    "tags": [],
                    "command": [missing],
                    "required_commands": [missing],
                }
            ]
        )

        completed = self.invoke(
            "run", "--no-module", "--results-dir", str(self.results)
        )

        self.assertEqual(completed.returncode, 3, completed.stdout + completed.stderr)
        self.assertIn("all were skipped", completed.stderr)
        self.assertEqual(self.read_only_summary()["totals"]["skip"], 1)

    def test_dry_run_does_not_execute_command(self):
        sentinel = self.work / "should-not-exist"
        self.write_manifest(
            [
                {
                    "id": "dry",
                    "suite": "unit",
                    "tags": [],
                    "command": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path({!r}).touch()".format(str(sentinel)),
                    ],
                }
            ]
        )

        completed = self.invoke(
            "run", "--no-module", "--dry-run", "--results-dir", str(self.results)
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertFalse(sentinel.exists())
        result = self.read_only_summary()["tests"][0]
        self.assertEqual(result["status"], "skip")
        self.assertIn("dry-run", result["reason"])

    def test_command_arguments_are_not_interpreted_by_a_shell(self):
        output = self.work / "argument.txt"
        injected = self.work / "injected"
        hostile_argument = "; touch {}; $(touch {})".format(injected, injected)
        self.write_manifest(
            [
                {
                    "id": "argv-safe",
                    "suite": "unit",
                    "tags": [],
                    "command": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                        str(output),
                        hostile_argument,
                    ],
                }
            ]
        )

        completed = self.invoke(
            "run", "--no-module", "--results-dir", str(self.results)
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), hostile_argument)
        self.assertFalse(injected.exists())

    def test_each_run_gets_a_unique_result_directory(self):
        self.write_manifest(
            [
                {
                    "id": "one",
                    "suite": "unit",
                    "tags": [],
                    "command": self.passing_command(),
                }
            ]
        )

        first = self.invoke("run", "--no-module", "--results-dir", str(self.results))
        second = self.invoke("run", "--no-module", "--results-dir", str(self.results))

        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        summaries = self.summaries()
        self.assertEqual(len(summaries), 2)
        self.assertNotEqual(summaries[0].parent, summaries[1].parent)

    @unittest.skipUnless(os.name == "posix", "signal/process-group behavior is POSIX-specific")
    def test_interrupt_stops_the_test_and_writes_a_summary(self):
        self.write_manifest(
            [
                {
                    "id": "long-running",
                    "suite": "unit",
                    "tags": [],
                    "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                },
                {
                    "id": "not-started",
                    "suite": "unit",
                    "tags": [],
                    "command": self.passing_command(),
                },
            ]
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(RUNNER),
                "--manifest",
                str(self.manifest),
                "run",
                "--no-module",
                "--results-dir",
                str(self.results),
            ],
            cwd=str(self.work),
            env={**os.environ, "FRAMEWORKS_TEST_XPUS": "0"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIsNotNone(process.stdout)
        output_lines = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
                if "[RUN    ] long-running" in line:
                    break
            elif process.poll() is not None:
                break
        else:
            process.kill()
            self.fail("runner did not start the long-running test")
        time.sleep(0.2)
        process.send_signal(signal.SIGINT)
        stdout_tail, stderr = process.communicate(timeout=8)

        self.assertEqual(process.returncode, 130, "".join(output_lines) + stdout_tail + stderr)
        summary = self.read_only_summary()
        self.assertTrue(summary["interrupted"])
        statuses = {item["id"]: item["status"] for item in summary["tests"]}
        self.assertEqual(statuses, {"long-running": "fail", "not-started": "skip"})

    def test_string_commands_are_rejected(self):
        self.write_manifest(
            [
                {
                    "id": "unsafe",
                    "suite": "unit",
                    "tags": [],
                    "command": "echo this must not be passed to a shell",
                }
            ]
        )

        completed = self.invoke("list")

        self.assertEqual(completed.returncode, 2)
        self.assertIn("command: must be an array of strings", completed.stderr)


if __name__ == "__main__":
    unittest.main()
