import os 

def get_filename_base(
    obstacle_config: str,
    medium_config: str,
    numerical_config: str
) -> str:
    """Get the base filename (no extension) of an
    obstacle configuration based on
    the config files used to create it.
    """
    obstacle_label = (
        os.path.splitext(
            os.path.basename(obstacle_config)
        )[0]
    )
    medium_label = (
        os.path.splitext(
            os.path.basename(medium_config)
        )[0]
    )
    numerical_label = (
        os.path.splitext(
            os.path.basename(numerical_config)
        )[0]
    )
    return f"scattering_{obstacle_label}_{medium_label}_{numerical_label}"