import time
from . import logger


# def periodic_bc(u: np.ndarray) -> np.ndarray:
#     """
#     Enforce periodic boundary conditions on the input data.
#     """
#     for axis in range(u.ndim):
#         first_slice = np.take(u, indices=0, axis=axis)
#         u = np.concatenate((u, np.expand_dims(first_slice, axis=axis)), axis=axis)
#     return u


def format_time(seconds: float, format='dd-hh:mm:ss') -> str:
    """
    Format the time into various formats: 'dd-hh:mm:ss', 'hh:mm:ss', or 'mm:ss'.

    Parameters:
        seconds (float): Time in seconds.
        format (str): Desired format ('dd-hh:mm:ss', 'hh:mm:ss', 'mm:ss').

    Returns:
        str: Formatted time string.
    """
    days, remainder = divmod(seconds, 86400)  # Total seconds in a day
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    format = format.casefold()
    if format == 'ss':
        return f"{seconds:02d}"
    elif format == 'mm:ss':
        return f"{int(minutes):02d}:{int(secs):02d}"
    elif format == 'hh:mm:ss':
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    elif format == 'dd-hh:mm:ss':
        return f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    else:
        raise ValueError("Invalid format. Choose 'dd-hh:mm:ss', 'hh:mm:ss', 'mm:ss', or 'ss'.")


class Timer:
    """Class to measure the time taken for a simulation to run.

    Parameters
    ----------
    verbose (bool)
        If True, prints timing information.
    """

    def __init__(self, verbose: bool = False, check_interval: int = 10e100):
        self.verbose = verbose
        self.check_interval = check_interval

        self.start_time = time.time()
        self.t0 = self.start_time

    def __call__(self, simulation_time: float, step: int):
        """
        Measure the time since the last call and print if verbose.

        Parameters:
            simulation_time (float): The current simulation time.
            step (int): The current simulation step.
        """
        if step % self.check_interval == 0:
            t1 = time.time()
            dt = t1 - self.t0
            self.t0 = t1

            if self.verbose:
                logger.info(f"Step = {step:08d}, time = {simulation_time:.2e}, runtime since last check = {format_time(dt, 'mm:ss')}")

    def start(self):
        """Print the start time of the simulation."""
        logger.info(f"Simulation started")

    def final(self):
        """Print the final timing information."""
        runtime = time.time() - self.start_time

        logger.info(f"Simulation completed. Total run time: {format_time(runtime, 'hh:mm:ss')}")