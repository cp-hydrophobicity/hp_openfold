import logging, sys

def configure_logging(level=logging.DEBUG):
    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    h_out = logging.StreamHandler(sys.stdout)
    h_out.setLevel(logging.DEBUG)
    h_out.addFilter(lambda r: r.levelno <= logging.INFO)
    h_out.setFormatter(fmt)
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)
    h_err.setFormatter(fmt)

    root.addHandler(h_out)
    root.addHandler(h_err)

    logging.captureWarnings(True)
    
    return logging.getLogger(__name__)
