import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
CPU_BIND_HELPER = REPOSITORY / "scripts" / "get_cpu_bind_aurora.sh"


class CpuBindingHelperTestCase(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run(
            ["bash", str(CPU_BIND_HELPER), *arguments],
            cwd=str(REPOSITORY),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )

    @staticmethod
    def binding_groups(output):
        prefix = "--cpu-bind list:"
        if not output.startswith(prefix):
            raise AssertionError("unexpected helper output: {!r}".format(output))
        return output.strip()[len(prefix) :].split(":")

    def test_twelve_rank_binding_stays_on_usable_cores(self):
        completed = self.invoke("12", "3")

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        groups = self.binding_groups(completed.stdout)
        self.assertEqual(len(groups), 12)
        self.assertNotIn("52", ",".join(groups).replace("-", ",").split(","))

    def test_out_of_range_shift_is_rejected(self):
        completed = self.invoke("12", "4")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("exceeds the maximum 3", completed.stderr)

    def test_invalid_rank_counts_are_rejected(self):
        for arguments in (("0",), ("not-a-number",), ("205", "--logical")):
            with self.subTest(arguments=arguments):
                completed = self.invoke(*arguments)
                self.assertNotEqual(completed.returncode, 0)

    def test_logical_binding_has_one_group_per_rank(self):
        completed = self.invoke("204", "--logical")

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(len(self.binding_groups(completed.stdout)), 204)


if __name__ == "__main__":
    unittest.main()
