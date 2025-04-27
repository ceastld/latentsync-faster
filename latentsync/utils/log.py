import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
