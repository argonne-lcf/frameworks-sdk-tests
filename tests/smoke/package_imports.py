#!/usr/bin/env python3
"""Import framework packages and report their installed versions.

The test intentionally imports each package in-process: binary/ABI conflicts are
one of the failures this suite is meant to catch after loading ``frameworks``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from typing import Iterable


DEFAULT_MODULES = (
    "numpy",
    "torch",
    "torchvision",
    "mpi4py",
    "dpctl",
    "dpnp",
)


def _version(module_name: str, module: object) -> str:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    distribution = module_name.replace("_", "-")
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def import_modules(names: Iterable[str]) -> None:
    failures: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except BaseException as error:  # imports may raise non-Exception ABI errors
            failures.append(f"{name}: {type(error).__name__}: {error}")
            print(f"FAIL {failures[-1]}", flush=True)
        else:
            location = getattr(module, "__file__", "built-in")
            print(f"PASS {name} {_version(name, module)} ({location})", flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} package import(s) failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modules",
        nargs="*",
        default=DEFAULT_MODULES,
        help="Import names (defaults to the core Frameworks SDK packages)",
    )
    args = parser.parse_args()
    import_modules(args.modules)


if __name__ == "__main__":
    main()
