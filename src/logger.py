from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_path=None, echo=False):
        self.log_path = Path(log_path) if log_path is not None else None
        self.echo = echo
        self.records = []

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step=None, **metrics):
        """
        Logs the given metrics.

        Args:
            step (int, optional): The current step or epoch. Useful for tracking.
            **metrics: Arbitrary keyword arguments representing metric names and values.
        """
        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'step': step,
            **metrics,
        }
        self.records.append(record)

        prefix = f"[Step {step}] " if step is not None else ""
        metric_str = " | ".join(f"{k}: {v}" for k, v in metrics.items())
        line = prefix + metric_str

        if self.log_path is not None:
            with self.log_path.open('a', encoding='utf-8') as log_file:
                log_file.write(f"{record['timestamp']} {line}\n")

        if self.echo:
            print(line)
