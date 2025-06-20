import numpy as np
import mrcfile
import glob
import os


def load_mrc(filename: str) -> np.ndarray:
    """
    Load the data in an mrc file into a numpy array.

    Parameters
    ----------
    filename: str, path to mrc file

    Returns
    -------
    np.ndarray, image or volume
    """
    with mrcfile.open(filename, "r", permissive=True) as mrc:
        return mrc.data


def save_mrc(
    data: np.ndarray,
    filename: str,
    apix: float = None,
    overwrite: bool = True,
):
    """
    Save a numpy array to mrc format.

    Parameters
    ----------
    data: np.ndarray, image or volume
    filename: str, save path
    apix: float, pixel size in Angstrom
    overwrite: bool, overwrite filename if already exists
    """
    if data.dtype != np.dtype("float32"):
        data = data.astype(np.float32)
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if apix:
            mrc.voxel_size = apix


def get_voxel_size(
    filename: str,
    isotropic: bool = True,
) -> float:
    """
    Extract voxel size from mrc file.

    Parameters
    ----------
    filename: str, path to mrc file
    isotropic: bool, extract single value assuming isotropic pixel size

    Returns
    -------
    apix: float, pixel size in Angstrom
    """
    apix = mrcfile.mmap(filename).voxel_size.tolist()
    if isotropic:
        return apix[0]
    return apix


