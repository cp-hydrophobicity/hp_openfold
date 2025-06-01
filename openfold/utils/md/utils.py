"""General utilities for molecular dynamics simulations."""

import os
import logging
import contextlib
import tempfile
from pathlib import Path

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# add the handler to the logger
logger.addHandler(ch)


@contextlib.contextmanager
def work_dir(output_dir: str, prefix: str = 'md_work_'):
    """Context manager for creating and managing a working directory.
    
    Args:
        output_dir: Base output directory where work directory will be created
        prefix: Prefix for the work directory name
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    work_path = Path(output_dir) / f"{prefix}{next(tempfile._get_candidate_names())}"
    work_path.mkdir(parents=True, exist_ok=True)
    
    try:
        yield work_path
    finally:
        if os.getenv('KEEP_WORK_FILES', '').lower() != 'true':
            for file in work_path.glob('*'):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove work file {file}: {e}")
            try:
                work_path.rmdir()
            except Exception as e:
                logger.warning(f"Failed to remove work directory {work_path}: {e}")
        else:
            logger.info(f"Keeping work files in {work_path}")
