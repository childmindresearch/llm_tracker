"""Access the synthetic practice data bundled with LLM Tracker."""

from importlib.resources import files
from pathlib import Path


def copy_demo_data(destination: str | Path = "tutorial_data") -> Path:
    """Copy practice files to a working folder and return its absolute path.

    Identical files are left untouched so notebook setup can be rerun safely.
    A differing existing file raises FileExistsError rather than overwriting
    edits. Choose another destination to obtain a fresh copy.
    """
    destination = Path(destination).expanduser().resolve()
    resources = files("llm_tracker").joinpath("_demo_data")
    contents = {
        resource.name: resource.read_bytes()
        for resource in resources.iterdir()
        if resource.is_file()
    }
    for name, content in contents.items():
        target = destination / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(
                f"Practice file differs from the bundled data: {target}. "
                "Choose another destination to keep your edits."
            )

    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        target = destination / name
        if not target.exists():
            with target.open("xb") as output:
                output.write(content)
    return destination
