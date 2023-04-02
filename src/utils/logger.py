import sys


class Logger(object):
    """Logger"""

    def __init__(self, logfile: str, overwrite_log: bool = False) -> None:
        self.terminal = sys.stdout
        self.logfile = logfile
        self.log = []
        if overwrite_log:
            open(logfile, "w").close()
        else:
            with open(logfile) as infile:
                self.log = [l for l in infile.readlines()]

    def save(self) -> None:
        with open(self.logfile, "w", encoding="utf-8") as log:
            log.writelines(self.log)

    def write(self, message: str) -> None:
        """Write in the terminal and save in log file"""
        self.terminal.write(message)
        self.log.append(message + "\n")

    def add_to_log(self, message: str) -> None:
        if message:
            self.log.append(message + "\n")

    def flush(self) -> None:
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
