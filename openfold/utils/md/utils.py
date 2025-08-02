import contextlib
import tempfile
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def work_dir(output_dir: str, prefix: str = 'md_work_'):
    """Context manager for creating and managing a working directory.
    
    Args:
        output_dir: Base output directory
        prefix: Prefix for the work directory name
    
    Returns:
        None
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