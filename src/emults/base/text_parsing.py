from typing import Optional
import os 
import sys
import contextlib

def get_filename_base(
    file: str
) -> str:
    """Gets just a file's name without any attached folders
    or file extensions"""
    return (
        os.path.splitext(
            os.path.basename(file)
        )[0]
    )

def get_full_configuration_filename_base(
    obstacle_config: str,
    medium_config: str,
    numerical_config: str,
    reference_config: Optional[str] = None
) -> str:
    """Get the base filename (no extension) of an
    obstacle configuration based on
    the config files used to create it.
    """
    obstacle_label = get_filename_base(obstacle_config)
    medium_label = get_filename_base(medium_config)
    numerical_label = get_filename_base(numerical_config)
    reference_label = get_filename_base(reference_config) if reference_config is not None else None 

    if reference_label is not None:
        return f"scattering_{obstacle_label}_{medium_label}_{numerical_label}_REFERENCE_{reference_label}"
    else:
        return f"scattering_{obstacle_label}_{medium_label}_{numerical_label}_NOREFERENCE"

@contextlib.contextmanager
def print_redirector(file_path=None, mode='w'):
    """
    A context manager to redirect print statements to a file or stdout.

    Args:
        file_path (str, optional): The path to the file for output.
                                   If None, output goes to stdout.
    """
    original_stdout = sys.stdout
    file_handle = None
    try:
        if file_path:
            file_handle = open(file_path, mode)
            sys.stdout = file_handle
        yield
    finally:
        sys.stdout = original_stdout
        if file_handle:
            file_handle.close()