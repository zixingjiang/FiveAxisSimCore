import sys
import os
import datetime


class Logger:
    COLORS = {
        "INFO": "\033[92m",   # green
        "WARN": "\033[93m",   # yellow
        "ERROR": "\033[91m",  # red
        "RESET": "\033[0m",
    }

    def __init__(self, name="FiveAxisSimCore", logfile=None, use_color=True):
        self.name = name
        self.use_color = use_color
        self.logfile = logfile
        if logfile:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            self._fh = open(logfile, "a", encoding="utf-8")
        else:
            self._fh = None

    def _timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _emit(self, level, msg):
        ts = self._timestamp()
        prefix = f"[{ts}] [{self.name}] [{level}] "
        text = f"{prefix}{msg}"

        # Console output
        if self.use_color and level in self.COLORS:
            color = self.COLORS[level]
            reset = self.COLORS["RESET"]
            sys.stdout.write(color + text + reset + "\n")
        else:
            sys.stdout.write(text + "\n")

        # File output
        if self._fh:
            self._fh.write(text + "\n")
            self._fh.flush()

    def info(self, msg): self._emit("INFO", msg)
    def warn(self, msg): self._emit("WARN", msg)
    def error(self, msg): self._emit("ERROR", msg)

    def close(self):
        if self._fh:
            self._fh.close()

    def __del__(self):
        if self._fh and not self._fh.closed:
            self._fh.close()


# ============================
# Self-test
# ============================
if __name__ == "__main__":
    log = Logger(logfile="logs/test.log")
    log.info("System initialized.")
    log.warn("Low voxel resolution may cause rough surface.")
    log.error("Simulation failed at step 42.")
    log.close()
