#!/usr/bin/env python3
"""Manifest-driven test runner for the frameworks SDK validation suite.

The runner intentionally depends only on the Python standard library.  Test
commands are argv arrays and are always executed without a shell.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import fnmatch
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EXIT_OK = 0
EXIT_TEST_FAILURE = 1
EXIT_CONFIGURATION = 2
EXIT_ENVIRONMENT = 3
EXIT_INTERRUPTED = 130

SUMMARY_VERSION = 1
ENV_MARKER = b"\0__FRAMEWORKS_TEST_ENV_BEGIN__\0"


class ManifestError(ValueError):
    """Raised when the suite manifest is invalid."""


class ModuleLoadError(RuntimeError):
    """Raised when the requested environment module cannot be loaded."""


class CaseInterrupted(KeyboardInterrupt):
    """Carries the result for a test interrupted by the user."""

    def __init__(self, result: Dict[str, Any]) -> None:
        super().__init__()
        self.result = result


@dataclass(frozen=True)
class TestCase:
    id: str
    suite: str
    tags: Tuple[str, ...]
    command: Tuple[str, ...]
    cwd: str
    env: Mapping[str, str]
    timeout: float
    required_commands: Tuple[str, ...]
    required_python_modules: Tuple[str, ...]
    min_xpus: int
    min_nodes: int
    enabled: bool
    skip_reason: str


@dataclass(frozen=True)
class Capacity:
    xpus: int
    nodes: int
    xpus_source: str
    nodes_source: str


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _field_error(location: str, message: str) -> ManifestError:
    return ManifestError("{}: {}".format(location, message))


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _field_error(location, "must be an object")
    return value


def _string_list(value: Any, location: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise _field_error(location, "must be an array of strings")
    result: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise _field_error("{}[{}]".format(location, index), "must be a non-empty string")
        result.append(item)
    return tuple(result)


def _nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _field_error(location, "must be a non-negative integer")
    return value


def _positive_timeout(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _field_error(location, "must be a positive number")
    result = float(value)
    if result <= 0:
        raise _field_error(location, "must be a positive number")
    return result


def _environment(value: Any, location: str) -> Dict[str, str]:
    mapping = _require_mapping(value, location)
    result: Dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise _field_error(location, "contains an invalid environment variable name")
        if isinstance(item, str):
            converted = item
        elif isinstance(item, bool):
            converted = "true" if item else "false"
        elif isinstance(item, (int, float)):
            converted = str(item)
        else:
            raise _field_error(
                "{}.{}".format(location, key), "must be a string, number, or boolean"
            )
        if "\0" in converted:
            raise _field_error("{}.{}".format(location, key), "must not contain NUL")
        result[key] = converted
    return result


def _requirement_value(
    case: Mapping[str, Any],
    case_requirements: Mapping[str, Any],
    defaults: Mapping[str, Any],
    default_requirements: Mapping[str, Any],
    name: str,
    fallback: Any,
) -> Any:
    for source in (case, case_requirements, defaults, default_requirements):
        if name in source:
            return source[name]
    return fallback


def load_manifest(path: Path) -> Tuple[Mapping[str, Any], List[TestCase]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError as error:
        raise ManifestError("manifest not found: {}".format(path)) from error
    except json.JSONDecodeError as error:
        raise ManifestError(
            "{}:{}:{}: invalid JSON: {}".format(path, error.lineno, error.colno, error.msg)
        ) from error
    except OSError as error:
        raise ManifestError("cannot read manifest {}: {}".format(path, error)) from error

    root = _require_mapping(document, "manifest")
    schema_version = root.get("schema_version", 1)
    if schema_version != 1:
        raise _field_error("schema_version", "only version 1 is supported")
    default_suites = _string_list(root.get("default_suites", []), "default_suites")
    if len(set(default_suites)) != len(default_suites):
        raise _field_error("default_suites", "must not contain duplicates")

    defaults = _require_mapping(root.get("defaults", {}), "defaults")
    default_requirements = _require_mapping(
        defaults.get("requirements", {}), "defaults.requirements"
    )
    default_env = _environment(defaults.get("env", {}), "defaults.env")
    default_timeout = _positive_timeout(defaults.get("timeout", 300), "defaults.timeout")

    raw_tests = root.get("tests", [])
    if not isinstance(raw_tests, list):
        raise _field_error("tests", "must be an array")

    tests: List[TestCase] = []
    seen_ids = set()
    for index, raw_value in enumerate(raw_tests):
        location = "tests[{}]".format(index)
        raw = _require_mapping(raw_value, location)

        test_id = raw.get("id")
        if not isinstance(test_id, str) or not test_id.strip():
            raise _field_error("{}.id".format(location), "must be a non-empty string")
        if test_id in seen_ids:
            raise _field_error("{}.id".format(location), "duplicate test id {!r}".format(test_id))
        seen_ids.add(test_id)

        suite = raw.get("suite", "default")
        if not isinstance(suite, str) or not suite.strip():
            raise _field_error("{}.suite".format(location), "must be a non-empty string")

        tags = _string_list(raw.get("tags", []), "{}.tags".format(location))
        if len(set(tags)) != len(tags):
            raise _field_error("{}.tags".format(location), "must not contain duplicates")

        command = _string_list(raw.get("command"), "{}.command".format(location))
        if not command:
            raise _field_error("{}.command".format(location), "must not be empty")
        if any("\0" in argument for argument in command):
            raise _field_error("{}.command".format(location), "arguments must not contain NUL")

        cwd = raw.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip() or "\0" in cwd:
            raise _field_error("{}.cwd".format(location), "must be a non-empty path string")

        case_env = dict(default_env)
        case_env.update(_environment(raw.get("env", {}), "{}.env".format(location)))
        timeout = _positive_timeout(
            raw.get("timeout", default_timeout), "{}.timeout".format(location)
        )

        case_requirements = _require_mapping(
            raw.get("requirements", {}), "{}.requirements".format(location)
        )
        required_commands = _string_list(
            _requirement_value(
                raw,
                case_requirements,
                defaults,
                default_requirements,
                "required_commands",
                [],
            ),
            "{}.required_commands".format(location),
        )
        required_python_modules = _string_list(
            _requirement_value(
                raw,
                case_requirements,
                defaults,
                default_requirements,
                "required_python_modules",
                [],
            ),
            "{}.required_python_modules".format(location),
        )
        min_xpus = _nonnegative_int(
            _requirement_value(
                raw, case_requirements, defaults, default_requirements, "min_xpus", 0
            ),
            "{}.min_xpus".format(location),
        )
        min_nodes = _nonnegative_int(
            _requirement_value(
                raw, case_requirements, defaults, default_requirements, "min_nodes", 0
            ),
            "{}.min_nodes".format(location),
        )

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise _field_error("{}.enabled".format(location), "must be a boolean")
        skip_reason = raw.get("skip_reason", "disabled in manifest")
        if not isinstance(skip_reason, str) or not skip_reason.strip():
            raise _field_error("{}.skip_reason".format(location), "must be a non-empty string")

        tests.append(
            TestCase(
                id=test_id,
                suite=suite,
                tags=tags,
                command=command,
                cwd=cwd,
                env=case_env,
                timeout=timeout,
                required_commands=required_commands,
                required_python_modules=required_python_modules,
                min_xpus=min_xpus,
                min_nodes=min_nodes,
                enabled=enabled,
                skip_reason=skip_reason,
            )
        )

    return root, tests


def select_tests(tests: Iterable[TestCase], args: argparse.Namespace) -> List[TestCase]:
    suite_patterns = args.suites or []
    tag_patterns = args.tags or []
    id_patterns = args.ids or []

    selected: List[TestCase] = []
    for case in tests:
        if suite_patterns and not any(
            fnmatch.fnmatchcase(case.suite, pattern) for pattern in suite_patterns
        ):
            continue
        if id_patterns and not any(
            fnmatch.fnmatchcase(case.id, pattern) for pattern in id_patterns
        ):
            continue
        if tag_patterns and not any(
            fnmatch.fnmatchcase(tag, pattern)
            for tag in case.tags
            for pattern in tag_patterns
        ):
            continue
        selected.append(case)
    return selected


def has_selection_filters(args: argparse.Namespace) -> bool:
    return bool(args.suites or args.tags or args.ids)


def apply_default_selection(
    args: argparse.Namespace, manifest: Mapping[str, Any]
) -> None:
    """Apply safe manifest defaults to execution commands, but never to list."""
    args.selection_source = "selectors" if has_selection_filters(args) else "all"
    if args.action not in ("run", "doctor"):
        return
    if args.all_tests:
        args.selection_source = "explicit --all"
        return
    if has_selection_filters(args):
        return
    default_suites = manifest.get("default_suites", [])
    if default_suites:
        args.suites = list(default_suites)
        args.selection_source = "manifest default_suites"


def load_module_environment(
    module_name: Optional[str], base_environment: Mapping[str, str]
) -> Tuple[Dict[str, str], str]:
    if module_name is None:
        return dict(base_environment), "module loading disabled"

    bash = shutil.which("bash", path=base_environment.get("PATH"))
    if not bash:
        raise ModuleLoadError("bash was not found in PATH")

    # The module name is $1 rather than interpolated into the script.  The
    # script itself is constant, so even a hostile module name cannot become
    # shell syntax.
    script = 'module load "$1" >/dev/null && printf "\\0__FRAMEWORKS_TEST_ENV_BEGIN__\\0" && env -0'
    try:
        completed = subprocess.run(
            [bash, "--login", "-c", script, "frameworks-test-harness", module_name],
            env=dict(base_environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ModuleLoadError(
            "timed out while loading module {!r}".format(module_name)
        ) from error
    except OSError as error:
        raise ModuleLoadError("could not start login shell: {}".format(error)) from error

    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        detail = stderr or "module command exited with status {}".format(completed.returncode)
        raise ModuleLoadError("could not load module {!r}: {}".format(module_name, detail))

    marker_offset = completed.stdout.find(ENV_MARKER)
    if marker_offset < 0:
        raise ModuleLoadError(
            "module {!r} loaded, but the login shell did not return an environment".format(
                module_name
            )
        )

    encoded_environment = completed.stdout[marker_offset + len(ENV_MARKER) :]
    environment: Dict[str, str] = {}
    for entry in encoded_environment.split(b"\0"):
        if not entry:
            continue
        key, separator, value = entry.partition(b"=")
        if not separator:
            continue
        environment[key.decode("utf-8", errors="surrogateescape")] = value.decode(
            "utf-8", errors="surrogateescape"
        )
    if "PATH" not in environment:
        raise ModuleLoadError("loaded environment does not contain PATH")

    detail = "loaded module {!r} via bash login shell".format(module_name)
    if stderr:
        detail += " (module message: {})".format(stderr.replace("\n", " "))
    return environment, detail


def _parse_integer_count(value: str) -> Optional[int]:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return int(stripped)
    match = re.search(r"(?:^|:)\s*(\d+)\s*(?:\(|$)", stripped)
    if match:
        return int(match.group(1))
    return None


def probe_torch_xpu_count(environment: Mapping[str, str]) -> Optional[int]:
    """Ask the loaded Python environment for XPU capacity, without importing here."""
    interpreter = shutil.which("python", path=environment.get("PATH")) or shutil.which(
        "python3", path=environment.get("PATH")
    )
    if interpreter is None:
        return None
    probe = "import torch; print(int(torch.xpu.device_count()))"
    try:
        completed = subprocess.run(
            [interpreter, "-c", probe],
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        count = int(output.splitlines()[-1])
    except ValueError:
        return None
    return count if count >= 0 else None


def probe_dpctl_xpu_count(environment: Mapping[str, str]) -> Optional[int]:
    """Use Level Zero discovery when a broken PyTorch reports zero XPUs."""
    interpreter = shutil.which("python", path=environment.get("PATH")) or shutil.which(
        "python3", path=environment.get("PATH")
    )
    if interpreter is None:
        return None
    probe = (
        "import dpctl; "
        "print(len(dpctl.get_devices(backend='level_zero', device_type='gpu')))"
    )
    try:
        completed = subprocess.run(
            [interpreter, "-c", probe],
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        count = int(output.splitlines()[-1])
    except ValueError:
        return None
    return count if count >= 0 else None


def detect_capacity(
    environment: Mapping[str, str],
    xpus_override: Optional[int],
    nodes_override: Optional[int],
) -> Capacity:
    if xpus_override is not None:
        xpus = xpus_override
        xpus_source = "--available-xpus"
    else:
        xpus = 0
        xpus_source = "default"
        for name in ("FRAMEWORKS_TEST_XPUS", "XPU_COUNT", "SLURM_GPUS_ON_NODE"):
            raw = environment.get(name)
            if raw is None:
                continue
            count = _parse_integer_count(raw)
            if count is not None:
                xpus = count
                xpus_source = name
                break
        else:
            affinity = environment.get("ZE_AFFINITY_MASK")
            if affinity and affinity.strip() not in ("-1", "NoDevFiles"):
                xpus = len([item for item in affinity.split(",") if item.strip()])
                xpus_source = "ZE_AFFINITY_MASK"
            else:
                torch_count = probe_torch_xpu_count(environment)
                if torch_count is not None and torch_count > 0:
                    xpus = torch_count
                    xpus_source = "torch.xpu.device_count()"
                else:
                    dpctl_count = probe_dpctl_xpu_count(environment)
                    if dpctl_count is not None:
                        xpus = dpctl_count
                        xpus_source = "dpctl Level Zero discovery"
                    elif torch_count is not None:
                        xpus = torch_count
                        xpus_source = "torch.xpu.device_count()"

    if nodes_override is not None:
        nodes = nodes_override
        nodes_source = "--available-nodes"
    else:
        nodes = 1
        nodes_source = "local default"
        for name in (
            "PBS_NNODES",
            "PBS_NUM_NODES",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NNODES",
        ):
            raw = environment.get(name)
            if raw is None:
                continue
            count = _parse_integer_count(raw)
            if count is not None:
                nodes = count
                nodes_source = name
                break
        else:
            nodefile = environment.get("PBS_NODEFILE")
            if nodefile:
                try:
                    with Path(nodefile).open("r", encoding="utf-8") as handle:
                        hosts = {line.strip() for line in handle if line.strip()}
                except (OSError, UnicodeError):
                    hosts = set()
                if hosts:
                    nodes = len(hosts)
                    nodes_source = "PBS_NODEFILE"

    return Capacity(xpus=xpus, nodes=nodes, xpus_source=xpus_source, nodes_source=nodes_source)


def case_cwd(case: TestCase, manifest_directory: Path) -> Path:
    path = Path(case.cwd).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def case_environment(case: TestCase, base_environment: Mapping[str, str]) -> Dict[str, str]:
    environment = dict(base_environment)
    environment.update(case.env)
    return environment


def resolve_executable(command: str, cwd: Path, environment: Mapping[str, str]) -> Optional[str]:
    if os.sep in command or (os.altsep and os.altsep in command):
        candidate = Path(command)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return str(candidate)
        return None
    return shutil.which(command, path=environment.get("PATH"))


def _python_for_case(case: TestCase, cwd: Path, environment: Mapping[str, str]) -> Optional[str]:
    executable_name = Path(case.command[0]).name.lower()
    if executable_name.startswith("python"):
        resolved = resolve_executable(case.command[0], cwd, environment)
        if resolved:
            return resolved
    return shutil.which("python", path=environment.get("PATH")) or shutil.which(
        "python3", path=environment.get("PATH")
    )


def python_module_available(
    interpreter: str,
    module_name: str,
    environment: Mapping[str, str],
    cache: Dict[Tuple[str, str, str], Tuple[bool, str]],
) -> Tuple[bool, str]:
    cache_key = (interpreter, module_name, environment.get("PYTHONPATH", ""))
    if cache_key in cache:
        return cache[cache_key]
    probe = (
        "import importlib.util, sys\n"
        "try:\n"
        "    found = importlib.util.find_spec(sys.argv[1]) is not None\n"
        "except (ImportError, ModuleNotFoundError, ValueError):\n"
        "    found = False\n"
        "raise SystemExit(0 if found else 1)\n"
    )
    try:
        completed = subprocess.run(
            [interpreter, "-c", probe, module_name],
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        result = (False, "python module probe failed: {}".format(error))
    else:
        if completed.returncode == 0:
            result = (True, "")
        else:
            result = (False, "python module {!r} is unavailable".format(module_name))
    cache[cache_key] = result
    return result


def requirement_reasons(
    case: TestCase,
    cwd: Path,
    environment: Mapping[str, str],
    capacity: Capacity,
    python_cache: Dict[Tuple[str, str, str], Tuple[bool, str]],
) -> List[str]:
    if not case.enabled:
        return [case.skip_reason]

    reasons: List[str] = []
    if capacity.xpus < case.min_xpus:
        reasons.append("needs {} XPU(s), found {}".format(case.min_xpus, capacity.xpus))
    if capacity.nodes < case.min_nodes:
        reasons.append("needs {} node(s), found {}".format(case.min_nodes, capacity.nodes))

    for command in case.required_commands:
        if resolve_executable(command, cwd, environment) is None:
            reasons.append("required command not found: {}".format(command))

    if case.required_python_modules:
        interpreter = _python_for_case(case, cwd, environment)
        if interpreter is None:
            reasons.append("python interpreter not found for module checks")
        else:
            for module_name in case.required_python_modules:
                available, reason = python_module_available(
                    interpreter, module_name, environment, python_cache
                )
                if not available:
                    reasons.append(reason)
    return reasons


def hard_preflight_reasons(
    case: TestCase, cwd: Path, environment: Mapping[str, str]
) -> List[str]:
    reasons: List[str] = []
    if not cwd.is_dir():
        reasons.append("working directory does not exist: {}".format(cwd))
    elif (
        case.command[0] not in case.required_commands
        and resolve_executable(case.command[0], cwd, environment) is None
    ):
        reasons.append("test command not found or not executable: {}".format(case.command[0]))
    return reasons


def _safe_log_name(index: int, test_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_id).strip("._") or "test"
    return "{:03d}-{}.log".format(index, stem[:120])


def _write_log(log_path: Path, lines: Sequence[str]) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def make_result(
    case: TestCase,
    status: str,
    reason: str,
    cwd: Path,
    log_path: Path,
    started_at: str,
    duration: float,
    returncode: Optional[int],
) -> Dict[str, Any]:
    return {
        "id": case.id,
        "suite": case.suite,
        "tags": list(case.tags),
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "duration_seconds": round(duration, 6),
        "started_at": started_at,
        "command": list(case.command),
        "cwd": str(cwd),
        "log": str(log_path),
    }


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def execute_case(
    case: TestCase,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> Dict[str, Any]:
    started_at = utc_now()
    start = time.monotonic()
    with log_path.open("wb") as log:
        header = (
            "test: {}\n"
            "suite: {}\n"
            "cwd: {}\n"
            "command: {}\n"
            "started: {}\n"
            "{}\n"
        ).format(
            case.id,
            case.suite,
            cwd,
            json.dumps(list(case.command)),
            started_at,
            "-" * 72,
        )
        log.write(header.encode("utf-8", errors="replace"))
        log.flush()
        try:
            proc = subprocess.Popen(
                list(case.command),
                cwd=str(cwd),
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            duration = time.monotonic() - start
            reason = "could not start test: {}".format(error)
            log.write(("\n[harness] {}\n".format(reason)).encode("utf-8", errors="replace"))
            return make_result(
                case, "fail", reason, cwd, log_path, started_at, duration, None
            )

        try:
            returncode = proc.wait(timeout=case.timeout)
        except subprocess.TimeoutExpired:
            terminate_process(proc)
            duration = time.monotonic() - start
            reason = "exceeded timeout of {:g} seconds".format(case.timeout)
            log.write(("\n[harness] {}\n".format(reason)).encode("utf-8"))
            return make_result(
                case, "timeout", reason, cwd, log_path, started_at, duration, proc.returncode
            )
        except KeyboardInterrupt:
            terminate_process(proc)
            duration = time.monotonic() - start
            reason = "interrupted by user"
            log.write(("\n[harness] {}\n".format(reason)).encode("utf-8"))
            result = make_result(
                case, "fail", reason, cwd, log_path, started_at, duration, proc.returncode
            )
            raise CaseInterrupted(result)

    duration = time.monotonic() - start
    if returncode == 0:
        return make_result(case, "pass", "", cwd, log_path, started_at, duration, 0)
    return make_result(
        case,
        "fail",
        "exited with status {}".format(returncode),
        cwd,
        log_path,
        started_at,
        duration,
        returncode,
    )


def create_run_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(10):
        name = "{}-{}-{}".format(timestamp, os.getpid(), uuid.uuid4().hex[:8])
        run_directory = parent / name
        try:
            run_directory.mkdir()
        except FileExistsError:
            continue
        (run_directory / "logs").mkdir()
        (run_directory / "artifacts").mkdir()
        return run_directory.resolve()
    raise OSError("could not allocate a unique result directory under {}".format(parent))


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def totals(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"pass": 0, "fail": 0, "skip": 0, "timeout": 0}
    for result in results:
        status = result["status"]
        counts[status] += 1
    return counts


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "{:.0f}ms".format(seconds * 1000)
    return "{:.2f}s".format(seconds)


def print_case_result(result: Mapping[str, Any]) -> None:
    status = str(result["status"]).upper()
    line = "[{:<7}] {} ({})".format(
        status, result["id"], format_duration(float(result["duration_seconds"]))
    )
    if result.get("reason"):
        line += " - {}".format(result["reason"])
    print(line, flush=True)


def _manifest_path(argument: str) -> Path:
    return Path(argument).expanduser().resolve()


def command_list(args: argparse.Namespace, tests: Sequence[TestCase]) -> int:
    selected = select_tests(tests, args)
    if not selected:
        print("No tests matched." if has_selection_filters(args) else "Manifest contains no tests.")
        return EXIT_CONFIGURATION if has_selection_filters(args) else EXIT_OK

    id_width = max(2, max(len(case.id) for case in selected))
    suite_width = max(5, max(len(case.suite) for case in selected))
    print("{:<{}}  {:<{}}  {:<20}  {}".format("ID", id_width, "SUITE", suite_width, "TAGS", "COMMAND"))
    for case in selected:
        print(
            "{:<{}}  {:<{}}  {:<20}  {}".format(
                case.id,
                id_width,
                case.suite,
                suite_width,
                ",".join(case.tags) or "-",
                shlex.join(case.command),
            )
        )
    print("\n{} test(s) selected.".format(len(selected)))
    return EXIT_OK


def _resolve_runtime_environment(
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], str, Capacity]:
    environment, module_detail = load_module_environment(args.module, os.environ)
    capacity = detect_capacity(
        environment, args.available_xpus, args.available_nodes
    )
    return environment, module_detail, capacity


def command_doctor(
    args: argparse.Namespace, manifest_path: Path, tests: Sequence[TestCase]
) -> int:
    selected = select_tests(tests, args)
    if not selected and has_selection_filters(args):
        print("No tests matched the selection filters.", file=sys.stderr)
        return EXIT_CONFIGURATION
    try:
        environment, module_detail, capacity = _resolve_runtime_environment(args)
    except ModuleLoadError as error:
        print("Environment error: {}".format(error), file=sys.stderr)
        return EXIT_ENVIRONMENT

    print("Manifest: {}".format(manifest_path))
    print("Environment: {}".format(module_detail))
    print(
        "Capacity: {} XPU(s) from {}; {} node(s) from {}".format(
            capacity.xpus,
            capacity.xpus_source,
            capacity.nodes,
            capacity.nodes_source,
        )
    )
    if not selected:
        print("Doctor: manifest contains no tests.")
        return EXIT_OK

    python_cache: Dict[Tuple[str, str, str], Tuple[bool, str]] = {}
    unavailable = 0
    for case in selected:
        cwd = case_cwd(case, manifest_path.parent)
        case_env = case_environment(case, environment)
        skip = requirement_reasons(case, cwd, case_env, capacity, python_cache)
        hard = hard_preflight_reasons(case, cwd, case_env)
        reasons = hard + skip
        if reasons:
            unavailable += 1
            print("[UNREADY] {} - {}".format(case.id, "; ".join(reasons)))
        else:
            print("[READY  ] {}".format(case.id))
    print(
        "Doctor: {} ready, {} unready.".format(len(selected) - unavailable, unavailable)
    )
    return EXIT_ENVIRONMENT if unavailable else EXIT_OK


def _results_parent(args: argparse.Namespace, manifest_path: Path) -> Path:
    if args.results_dir is None:
        return manifest_path.parent / "results"
    path = Path(args.results_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def command_run(
    args: argparse.Namespace, manifest_path: Path, tests: Sequence[TestCase]
) -> int:
    selected = select_tests(tests, args)
    if not selected:
        print("No tests matched; nothing was run.", file=sys.stderr)
        return EXIT_CONFIGURATION

    try:
        run_directory = create_run_directory(_results_parent(args, manifest_path))
    except OSError as error:
        print("Could not create result directory: {}".format(error), file=sys.stderr)
        return EXIT_CONFIGURATION

    started_at = utc_now()
    start = time.monotonic()
    summary: Dict[str, Any] = {
        "summary_version": SUMMARY_VERSION,
        "manifest": str(manifest_path),
        "result_directory": str(run_directory),
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "module": args.module,
        "loaded_modules": None,
        "dry_run": args.dry_run,
        "interrupted": False,
        "selection": {
            "suites": args.suites or [],
            "tags": args.tags or [],
            "ids": args.ids or [],
            "source": args.selection_source,
        },
        "capacity": None,
        "environment_detail": None,
        "harness_error": None,
        "totals": totals([]),
        "tests": [],
    }

    try:
        environment, module_detail, capacity = _resolve_runtime_environment(args)
    except KeyboardInterrupt:
        summary["interrupted"] = True
        summary["harness_error"] = "interrupted while preparing environment"
        summary["finished_at"] = utc_now()
        summary["duration_seconds"] = round(time.monotonic() - start, 6)
        write_summary(run_directory / "summary.json", summary)
        print("Interrupted. Summary: {}".format(run_directory / "summary.json"), file=sys.stderr)
        return EXIT_INTERRUPTED
    except ModuleLoadError as error:
        summary["harness_error"] = str(error)
        summary["finished_at"] = utc_now()
        summary["duration_seconds"] = round(time.monotonic() - start, 6)
        write_summary(run_directory / "summary.json", summary)
        print("Environment error: {}".format(error), file=sys.stderr)
        print("Summary: {}".format(run_directory / "summary.json"), file=sys.stderr)
        return EXIT_ENVIRONMENT

    summary["environment_detail"] = module_detail
    summary["loaded_modules"] = [
        item
        for item in environment.get("LOADEDMODULES", "").split(":")
        if item
    ]
    summary["capacity"] = {
        "xpus": capacity.xpus,
        "nodes": capacity.nodes,
        "xpus_source": capacity.xpus_source,
        "nodes_source": capacity.nodes_source,
    }
    print("Results: {}".format(run_directory), flush=True)
    print("Environment: {}".format(module_detail), flush=True)
    if args.dry_run:
        print("Dry run: commands will be checked but not executed.", flush=True)

    python_cache: Dict[Tuple[str, str, str], Tuple[bool, str]] = {}
    results: List[Dict[str, Any]] = []
    interrupted = False
    for index, case in enumerate(selected, start=1):
        cwd = case_cwd(case, manifest_path.parent)
        env = case_environment(case, environment)
        log_path = run_directory / "logs" / _safe_log_name(index, case.id)
        artifact_name = _safe_log_name(index, case.id)[:-4]
        artifact_directory = run_directory / "artifacts" / artifact_name
        artifact_directory.mkdir()
        env.setdefault("FRAMEWORKS_TEST_RUN_DIR", str(run_directory))
        env.setdefault("FRAMEWORKS_TEST_ARTIFACT_DIR", str(artifact_directory))
        env.setdefault("MASTER_PORT", str(20000 + ((os.getpid() * 97 + index) % 40000)))
        instant_start = utc_now()
        try:
            requirement_issues = requirement_reasons(
                case, cwd, env, capacity, python_cache
            )
            hard_issues = hard_preflight_reasons(case, cwd, env)
        except KeyboardInterrupt:
            reason = "interrupted by user during preflight"
            _write_log(
                log_path,
                ["test: {}".format(case.id), "status: fail", "reason: {}".format(reason)],
            )
            result = make_result(
                case, "fail", reason, cwd, log_path, instant_start, 0.0, None
            )
            interrupted = True
        else:
            if hard_issues:
                reason = "; ".join(hard_issues)
                _write_log(log_path, ["test: {}".format(case.id), "status: fail", "reason: {}".format(reason)])
                result = make_result(
                    case, "fail", reason, cwd, log_path, instant_start, 0.0, None
                )
            elif requirement_issues:
                reason = "; ".join(requirement_issues)
                _write_log(log_path, ["test: {}".format(case.id), "status: skip", "reason: {}".format(reason)])
                result = make_result(
                    case, "skip", reason, cwd, log_path, instant_start, 0.0, None
                )
            elif args.dry_run:
                reason = "dry-run: would execute {}".format(shlex.join(case.command))
                _write_log(log_path, ["test: {}".format(case.id), "status: skip", "reason: {}".format(reason)])
                result = make_result(
                    case, "skip", reason, cwd, log_path, instant_start, 0.0, None
                )
            else:
                try:
                    print("[RUN    ] {}".format(case.id), flush=True)
                    result = execute_case(case, cwd, env, log_path)
                except CaseInterrupted as error:
                    result = error.result
                    interrupted = True
                except KeyboardInterrupt:
                    reason = "interrupted by user before the test started"
                    _write_log(
                        log_path,
                        [
                            "test: {}".format(case.id),
                            "status: fail",
                            "reason: {}".format(reason),
                        ],
                    )
                    result = make_result(
                        case, "fail", reason, cwd, log_path, instant_start, 0.0, None
                    )
                    interrupted = True

        result["artifact_directory"] = str(artifact_directory)
        results.append(result)
        print_case_result(result)
        if interrupted:
            for remaining_index, remaining in enumerate(
                selected[index:], start=index + 1
            ):
                remaining_cwd = case_cwd(remaining, manifest_path.parent)
                remaining_log = (
                    run_directory / "logs" / _safe_log_name(remaining_index, remaining.id)
                )
                remaining_artifact = (
                    run_directory
                    / "artifacts"
                    / _safe_log_name(remaining_index, remaining.id)[:-4]
                )
                remaining_artifact.mkdir()
                reason = "not run because the harness was interrupted"
                _write_log(
                    remaining_log,
                    ["test: {}".format(remaining.id), "status: skip", "reason: {}".format(reason)],
                )
                skipped = make_result(
                    remaining,
                    "skip",
                    reason,
                    remaining_cwd,
                    remaining_log,
                    utc_now(),
                    0.0,
                    None,
                )
                skipped["artifact_directory"] = str(remaining_artifact)
                results.append(skipped)
                print_case_result(skipped)
            break

    summary["tests"] = results
    summary["totals"] = totals(results)
    summary["interrupted"] = interrupted
    summary["finished_at"] = utc_now()
    summary["duration_seconds"] = round(time.monotonic() - start, 6)
    write_summary(run_directory / "summary.json", summary)

    counts = summary["totals"]
    print(
        "\nSummary: {pass} passed, {fail} failed, {skip} skipped, "
        "{timeout} timed out in {duration}.".format(
            duration=format_duration(float(summary["duration_seconds"])), **counts
        )
    )
    print("Summary JSON: {}".format(run_directory / "summary.json"))

    if interrupted:
        return EXIT_INTERRUPTED
    if counts["fail"] or counts["timeout"]:
        return EXIT_TEST_FAILURE
    if not args.dry_run and counts["pass"] == 0:
        print(
            "No selected tests executed successfully; all were skipped.",
            file=sys.stderr,
        )
        return EXIT_ENVIRONMENT
    return EXIT_OK


def nonnegative_cli_int(value: str) -> int:
    try:
        converted = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if converted < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return converted


def add_manifest_argument(parser: argparse.ArgumentParser, child: bool = False) -> None:
    kwargs: Dict[str, Any] = {
        "metavar": "PATH",
        "help": "suite manifest (default: suite.json next to runner.py)",
    }
    if child:
        kwargs["default"] = argparse.SUPPRESS
    else:
        kwargs["default"] = str(Path(__file__).with_name("suite.json"))
    parser.add_argument("--manifest", **kwargs)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--suite",
        dest="suites",
        action="append",
        metavar="GLOB",
        help="select a suite by glob; repeat to OR patterns",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        metavar="GLOB",
        help="select tests with a matching tag glob; repeat to OR patterns",
    )
    parser.add_argument(
        "--id",
        dest="ids",
        action="append",
        metavar="GLOB",
        help="select a test id by glob; repeat to OR patterns",
    )


def add_execution_selection_arguments(parser: argparse.ArgumentParser) -> None:
    add_selection_arguments(parser)
    parser.add_argument(
        "--all",
        dest="all_tests",
        action="store_true",
        help="select every test, bypassing manifest default_suites",
    )


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    module_group = parser.add_mutually_exclusive_group()
    module_group.add_argument(
        "--module",
        default="frameworks",
        metavar="NAME",
        help="environment module to load (default: frameworks)",
    )
    module_group.add_argument(
        "--no-module",
        dest="module",
        action="store_const",
        const=None,
        help="run in the current environment without loading a module",
    )
    parser.add_argument(
        "--available-xpus",
        type=nonnegative_cli_int,
        metavar="N",
        help="override detected XPU capacity",
    )
    parser.add_argument(
        "--available-nodes",
        type=nonnegative_cli_int,
        metavar="N",
        help="override detected node capacity",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_tests",
        description="Run frameworks SDK validation tests from a JSON manifest.",
    )
    add_manifest_argument(parser)
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="list selected tests")
    add_manifest_argument(list_parser, child=True)
    add_selection_arguments(list_parser)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check module, capacity, and test prerequisites"
    )
    add_manifest_argument(doctor_parser, child=True)
    add_execution_selection_arguments(doctor_parser)
    add_environment_arguments(doctor_parser)

    run_parser = subparsers.add_parser("run", help="run selected tests")
    add_manifest_argument(run_parser, child=True)
    add_execution_selection_arguments(run_parser)
    add_environment_arguments(run_parser)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report commands without executing them",
    )
    run_parser.add_argument(
        "--results-dir",
        metavar="PATH",
        help="parent for unique run directories (default: MANIFEST_DIR/results)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = _manifest_path(args.manifest)
    try:
        manifest, tests = load_manifest(manifest_path)
    except ManifestError as error:
        print("Manifest error: {}".format(error), file=sys.stderr)
        return EXIT_CONFIGURATION

    if args.action in ("run", "doctor") and args.all_tests and has_selection_filters(args):
        parser.error("--all cannot be combined with --suite, --tag, or --id")
    apply_default_selection(args, manifest)

    try:
        if args.action == "list":
            return command_list(args, tests)
        if args.action == "doctor":
            return command_doctor(args, manifest_path, tests)
        if args.action == "run":
            return command_run(args, manifest_path, tests)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    parser.error("unknown command")
    return EXIT_CONFIGURATION


if __name__ == "__main__":
    raise SystemExit(main())
