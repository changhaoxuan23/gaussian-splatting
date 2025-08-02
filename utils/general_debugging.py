import faulthandler
from signal import SIGUSR1


def export_traceback(signal_number: int = SIGUSR1) -> None:
  faulthandler.register(signum=signal_number, chain=True)
