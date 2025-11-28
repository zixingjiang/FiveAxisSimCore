import re
import numpy as np
from typing import TextIO


class OutputSetting:
    """Define the order of G-code fields to output."""
    def __init__(self, word_entry=None):
        # Default field order: G, X, Y, Z, I, J, K
        self.word_entry = word_entry or ["G", "X", "Y", "Z", "I", "J", "K"]


class GCodeReader:
    """G-code reader that parses instructions with floating-point and scientific notation."""

    # Precompiled regex: capture command letter and its floating-point value (including scientific notation)
    pattern = re.compile(r"([A-Z])([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    @staticmethod
    def load(fp: TextIO, setting: OutputSetting=OutputSetting()) -> np.ndarray:
        """
        Read a G-code file and return a trajectory matrix.
        Each row corresponds to a G-code state.
        """
        prev = {word: 0.0 for word in setting.word_entry}  # Initialize previous state
        trajectory = []

        for line in fp:
            line = line.strip()
            if not line or line.startswith("("):  # Skip empty lines and comments
                continue

            # Match all commands
            for word, val in GCodeReader.pattern.findall(line):
                if word in prev:
                    prev[word] = float(val)

            # Generate a row of current state (excluding the first word, e.g., G)
            row = [prev[word] for word in setting.word_entry[1:]]
            trajectory.append(row)

        traj_arr = np.array(trajectory, dtype=float)
        return traj_arr


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    sample = """\
G1.00000000 X-6.83576298 Y45.80086991 Z3.37087794 I0.00000000 J0.00000000 K1.00000000
Y45.76093381 Z3.37287474
Y45.72099771 Z3.37487155
"""
    from io import StringIO
    traj = GCodeReader.load(StringIO(sample))
    print(traj)
