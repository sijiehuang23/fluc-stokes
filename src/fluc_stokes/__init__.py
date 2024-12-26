import logging

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError:
    raise ImportError("Rich library is required for logging.")


handler = RichHandler(
    console=Console(width=120),
    show_time=False,
    show_level=False,
    show_path=False
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[handler]
)
logger = logging.getLogger("steady_stokes")
