import numpy as np
from astropy.io.fits import (
    PrimaryHDU,
    ImageHDU,
    BinTableHDU,
    TableHDU,
    Header,
    HDUList,
)

from typing import Dict, Any, Optional,List


class HDUClone:
    """
    A picklable clone of astropy.io.fits HDU objects.
    
    Attributes:
    ----------
    data : np.ndarray or None
        The data contained in the HDU. Can be None for HDUs without data.
    header : Dict[str, Any]
        The header information as a dictionary.
    hdu_type : str
        The type of the HDU (e.g., 'PRIMARY', 'IMAGE', 'BINTABLE').
    """

    def __init__(
        self,
        data: Optional[np.ndarray],
        header: Dict[str, Any],
        hdu_type: str = "IMAGE",
    ):
        self.data = data
        self.header = header
        self.hdu_type = hdu_type.upper()  # Ensure consistency in type naming

    def to_hdu(self):
        """
        Convert the clone back to an astropy HDU object based on its type.
        
        Returns:
        -------
        astropy.io.fits.PrimaryHDU or ImageHDU or BinTableHDU or TableHDU or CompressedHDU or UndefinedHDU
            The corresponding astropy HDU object.
        
        Raises:
        ------
        ValueError:
            If the hdu_type is unrecognized.
        """
        hdu_type_upper = self.hdu_type.upper()

        if hdu_type_upper == "PRIMARY":
            return PrimaryHDU(data=self.data, header=self.header)
        elif hdu_type_upper == "IMAGE":
            return ImageHDU(data=self.data, header=self.header)
        elif hdu_type_upper == "BINTABLE":
            return BinTableHDU(data=self.data, header=self.header)
        elif hdu_type_upper == "ASCII_TABLE":
            return TableHDU(data=self.data, header=self.header)
        else:
            raise ValueError(f"Unknown HDU type: {self.hdu_type}")

    @staticmethod
    def from_hdu(hdu):
        """
        Create an HDUClone instance from an astropy HDU object.
        
        Parameters:
        ----------
        hdu : astropy.io.fits.PrimaryHDU or ImageHDU or BinTableHDU or TableHDU or CompressedHDU or UndefinedHDU
            The astropy HDU object to clone.
        
        Returns:
        -------
        HDUClone
            The cloned HDUClone object.
        """
        # Extract HDU type based on the class name
        hdu_type = type(hdu).__name__.upper().replace("HDU", "")

        # Convert header to a regular dictionary
        header_dict = (hdu.header)

        # Handle cases where HDU might not have data
        data = hdu.data.copy() if hdu.data is not None else None

        return HDUClone(data=data, header=header_dict, hdu_type=hdu_type)


class HDUListClone(list):
    """
    A picklable clone of astropy.io.fits HDUList objects.
    
    Inherits from the built-in list to leverage native list functionalities.
    
    Attributes:
    ----------
    Each item in the list is an instance of HDUClone.
    """

    def __init__(self, hdu_clones=None):
        """
        Initialize the HDUListClone.
        
        Parameters:
        ----------
        hdu_clones : Optional[List[HDUClone]], default None
            A list of HDUClone objects to initialize the HDUListClone.
        """
        if hdu_clones is None:
            hdu_clones = []
        super().__init__(hdu_clones)

    def to_hdulist(self):
        """
        Convert the clone list back to an astropy HDUList object.
        
        Returns:
        -------
        astropy.io.fits.HDUList
            The corresponding astropy HDUList object.
        """
        return HDUList([hdu.to_hdu() for hdu in self])

    @classmethod
    def from_hdulist(cls, hdulist: HDUList):
        """
        Create an HDUListClone from an astropy HDUList object.
        
        Parameters:
        ----------
        hdulist : astropy.io.fits.HDUList
            The astropy HDUList object to clone.
        
        Returns:
        -------
        HDUListClone
            The cloned HDUListClone object.
        """
        hdu_clones = [HDUClone.from_hdu(hdu) for hdu in hdulist]
        return cls(hdu_clones)

    def __repr__(self):
        """
        Get a string representation of the HDUListClone.
        
        Returns:
        -------
        str
            A string indicating the number of HDUs.
        """
        return f"HDUListClone({len(self)} HDUs)"
