import json
from typing import Dict, List,Tuple, Union
import pkg_resources
import numpy as np
from collections.abc import Iterable
import pandas as pd
from pathlib import Path

class LineCatalog:
    def __init__(self, file_location: str = None, verbose: int = 0) -> None:
        """
        Initializes the LineCatalog class.

        :param file_location: Path to the JSON file containing the line catalog data.
        :param verbose: Verbosity level. A higher value will print more details.
        """
        self.PATH = file_location
        self.verbose = verbose
        self.load()
        self._LINES = pd.DataFrame(self._CATALOGUE["LINES"])
        self._WINDOWS = pd.DataFrame(self._CATALOGUE["SPECTRAL_WINDOWS_CATALOGUE"])
        self.verbose = verbose

    def load(self) -> None:
        """
        Loads the line catalog data from the JSON file.
        """
        # import os
        # print("current working directory: "+os.getcwd())
        if self.PATH is None:
            self.PATH = pkg_resources.resource_filename(
                "saffron", "line_catalog/SPICE_SpecLines.json"
            )
        # with open(self.PATH, "r") as f:
        path__= Path(self.PATH).resolve()
        print(path__)
        with open(path__, "r") as f:
            if self.verbose >= 1:
                print("loading from ", self.PATH)
            self._CATALOGUE = json.load(f)

    def dump(self, new_path: str = "::SAME") -> None:
        """
        Dumps the line catalog and spectral windows catalog in a formatted JSON structure.

        :param new_path: Path to save the formatted catalog. If set to "::SAME", it will use the current file location.
        """
        if new_path == "::SAME":
            new_path = self.PATH

        # Prepare the data structure for dumping
        dumpable = {
            "LINES": [
                {"ID": int(line_id), "name": name, "wvl": wvl}
                for line_id, name, wvl in zip(self._LINES["ID"], self._LINES["name"], self._LINES["wvl"])
            ],
            "SPECTRAL_WINDOWS_CATALOGUE": [
                {"lines": [int(line_id) for line_id in lines], "max_line": int(max_line)}
                for lines, max_line in zip(self._WINDOWS["lines"], self._WINDOWS["max_line"])
            ]
        }

        # If there is a "Deleted" section, include it in the dump
        if "Deleted" in self._CATALOGUE:
            dumpable["Deleted"] = self._CATALOGUE["Deleted"]

        # Create the JSON with the required formatting
        with open(new_path, "w") as f:
            f.write("{\n")
            
            # Writing the "LINES" section
            f.write('  "LINES": [\n')
            for line in dumpable["LINES"]:
                f.write(
                    f'      {{"ID": {line["ID"]}, "name": "{line["name"]}", "wvl": {line["wvl"]:.2f}}}'
                    + (",\n" if line != dumpable["LINES"][-1] else "\n")
                )
            f.write('  ],\n')

            # Writing the "SPECTRAL_WINDOWS_CATALOGUE" section
            f.write('  "SPECTRAL_WINDOWS_CATALOGUE": [\n')
            for window in dumpable["SPECTRAL_WINDOWS_CATALOGUE"]:
                lines = ", ".join(map(str, window["lines"]))
                f.write(
                    f'      {{"lines": [{lines}], "max_line": {window["max_line"]}}}'
                    + (",\n" if window != dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1] else "\n")
                )
            f.write('  ],\n')

            # Writing the "Deleted" section if it exists
            if "Deleted" in dumpable:
                f.write('  "Deleted": [\n')
                for window in dumpable["Deleted"]:
                    lines = ", ".join(map(str, window["lines"]))
                    f.write(
                        f'      {{"lines": [{lines}], "max_line": {window["max_line"]}}}'
                        + (",\n" if window != dumpable["Deleted"][-1] else "\n")
                    )
                f.write('  ]\n')

            f.write("}\n")

    def _update_catalog(self) -> None:
        """
        Updates the line and spectral windows data in the catalog.
        """
        self._CATALOGUE["LINES"] = self._LINES.to_dict()
        self._CATALOGUE["SPECTRAL_WINDOWS_CATALOGUE"] = self._WINDOWS.to_dict()

    def add_window(self, lines: List[Tuple[str, float] | int], max_line: Tuple[str, float] | int) -> None:
        """
        Adds a new spectral window to the line catalog.

        :param lines: List of lines, either as [name, wavelength] tuples or line IDs.
        :param max_line: The maximum line in the window, either as [name, wavelength] or a line ID.
        """
        # Validate that the lines are all of the same type [name,wvl] or [ID]
        if (not np.all([isinstance(line,Iterable) for line in lines]) ) and (not np.all([isinstance(line,int) for line in lines]) ):
            raise ValueError(f"lines should be of the form [[name,wvl],...] or [ID,....]")
        if not isinstance(max_line[0],int) and not isinstance(max_line[0],Iterable):
            raise ValueError(f"max_line should be of the form [name,wvl] or [ID]")
        
        # Convert the lines to IDs
        if isinstance(lines[0],Iterable):
            lines_ID = []
            for line in lines :
                assert  self._check_valide_ion_name(line[0]), f"The name: {line} is not a valid ion name according to the CHIANTI format."
                try: 
                    lineID = self.line2ID(line[0],line[1])
                except: 
                    raise AssertionError(f"the line with name: {line[0]} and wvl: {line[1]} doesn't exist in the line catalog, please make sure you picked the right line or add it if it doesn't exist.\nline catalog: {self._LINES}")
                lines_ID.append(lineID)
        else:
            for line in lines:
                try: self.ID2line(line)
                except: raise ValueError(f"the line with ID: {line} doesn't exist in the line catalog, please make sure you picked the right line or add it if it doesn't exist")
            lines_ID = lines
        
        # Convert the max_line to ID
        if isinstance(max_line[0],Iterable):
            max_line_ID = self.line2ID(max_line[0],max_line[1])
        else:
            max_line_ID = max_line
        
        # Check if the max_line is in the lines
        assert max_line_ID in lines_ID, f"the max_line: {max_line} should be in the lines: {lines}"
        
        # Check if the lines are valid according to the CHIANTI format
        for line in lines:
            assert self._check_valide_ion_name(line[0]), f"The name: {line} is not a valid ion name according to the CHIANTI format."
            
        # Check if the exact window already exists (same lines and max_line)
        for window in self._WINDOWS.iterrows():
            if set(window[1]["lines"]) == set(lines_ID):
                raise ValueError(f"The window with lines '{lines}' already exists.")
        

        # Create a new DataFrame for the new window
        new_window_df = pd.DataFrame({"lines": [lines_ID], "max_line": [max_line_ID]})

        # Append the new window to the _WINDOWS DataFrame
        self._WINDOWS = pd.concat([self._WINDOWS, new_window_df], axis=0, ignore_index=True)
        
        # # Append the new window to the _WINDOWS DataFrame
        # self._WINDOWS = pd.concat([self._WINDOWS, new_window_df], axis=0, ignore_index=True)

        # Update the catalog data
        self._update_catalog()
    
    def add_line(self, name: str, wvl: float) -> None:
        """
        Adds a new line to the line catalog.

        :param name: Name of the new line.
        :param wvl: Wavelength of the new line.
        """
        # Validate the ion name using the CHIANTI format
        if not self._check_valide_ion_name(name):
            raise ValueError(f"The name: {name} is not a valid ion name according to the CHIANTI format.")

        # Check if the exact line already exists (same name and wavelength)
        existing_line = self._LINES[(self._LINES["name"] == name) & (self._LINES["wvl"] == wvl)]
        if not existing_line.empty:
            raise ValueError(f"The line with name '{name}' and wavelength '{wvl}' already exists.")

        # Determine the new ID by finding the smallest available integer
        existing_ids = set(self._LINES["ID"])
        new_id = 0
        while new_id in existing_ids:
            new_id += 1

        # Create a new DataFrame for the new line
        new_line_df = pd.DataFrame({"ID": [new_id], "name": [name], "wvl": [wvl]})

        # Append the new line to the _LINES DataFrame
        self._LINES = pd.concat([self._LINES, new_line_df], axis=0, ignore_index=True)

        # Sort the DataFrame by wavelength
        self._LINES = self._LINES.sort_values(by="wvl", ignore_index=True)

        # Update the catalog data
        self._update_catalog()
    
    def remove_window(self, lines: List[str]) -> None:
        """
        Removes a spectral window from the line catalog.

        :param lines: List of lines in the window to be removed, either as line IDs or [name, wavelength].
        """
        #turn lines from names to IDs if it is not already the case
        if    np.all([isinstance(line,int) for line in lines]): 
            lines_ID = lines
        elif  np.all([isinstance(line[0],str) and isinstance(line[1],float) for line in lines]):
            lines_ID = []
            for line in lines:
                if isinstance(line,Iterable):
                    assert  self._check_valide_ion_name(line[0]), f"The name: {line} is not a valid ion name according to the CHIANTI format."
                    try: 
                        lineID = self.line2ID(line[0],line[1])
                    except: 
                        raise AssertionError(f"the line with name: {line[0]} and wvl: {line[1]} doesn't exist in the line catalog, please make sure you picked the right line or add it if it doesn't exist.\nline catalog: {self._LINES}")
                    lines_ID.append(lineID)
        
        else: 
            raise ValueError(f"lines should be of the form [[name,wvl],...] or [ID,....]")


        found = self._WINDOWS.lines.apply(lambda x: set(x) == set(lines_ID))
        if not found.any():
            raise ValueError(
                f"the window to remove with this composition: {lines} doesn't exist "
            )
        else:
            self._WINDOWS = self._WINDOWS[np.logical_not(found)]
            self._update_catalog()

    def remove_line(self, name: str = None, wvl: float = None, ID: int = None) -> None:
        """
        Removes a line from the line catalog.

        :param name: Name of the line to be removed.
        :param wvl: Wavelength of the line to be removed.
        :param ID: ID of the line to be removed.
        """
        assert (name is not None and wvl is not None) or ID is not None, "you should provide either the name and wvl or the ID"
        if name is not None and wvl is not None:
            line_ID = self.line2ID(name=name,wvl=wvl)
        else:
            line_ID = ID
        
        found = self._LINES.ID.apply(lambda x: x == line_ID)
        windows_with_line_in = self._WINDOWS.lines.apply(lambda x: line_ID in x)
        assert found.any(), f"the line with ID: {line_ID} doesn't exist in the catalog"
        assert not windows_with_line_in.any(), f"The line:{line_ID} is in some windows\n{self._WINDOWS[windows_with_line_in]}\n you should delete them first"
        self._LINES = self._LINES[np.logical_not(found)]
        self._update_catalog()

    # getters
    def get_catalog(self) -> pd.DataFrame:
        """
        Retrieves the full catalog of lines and spectral windows.
        
        :return: The catalog as a pandas DataFrame.
        """
        return self._CATALOGUE

    def get_catalog_lines(self) -> pd.DataFrame:
        """
        Retrieves the catalog of spectral lines.
        
        :return: A pandas DataFrame of lines.
        """
        return self._LINES

    def get_catalog_windows(self, byID: bool = True) -> pd.DataFrame | dict:
        """
        Retrieves the spectral windows catalog.

        :param byID: If True, returns the catalog by line ID, otherwise by name and wavelength.
        :return: A pandas DataFrame or dictionary of spectral windows.
        """
        if byID:
            return self._WINDOWS
        else:
            window_value = []
            for window in self._WINDOWS.iterrows():
                lines= [self.ID2line(ID) for ID in window[1]["lines"]]
                max_line = self.ID2line(window[1]["max_line"])
                window_value.append({"lines":lines,"max_line":max_line})
            return (window_value) 
    
    # representations
    def show_catalog_windows(self) -> None:
        """
        Displays the spectral windows catalog in a readable format.
        """
        
        dic_value = self.get_catalog_windows(byID=False)
        for window in dic_value:
            
            print(f"lines: {' '.join([str(tuple(line.values())[1:]) for line in window['lines']])},\nmax_line: {tuple(window['max_line'].values())[1:]}")

    def _check_valide_ion_name(self, ion: str) -> bool:
        """
        Validates the ion name format according to CHIANTI standards.

        :param ion: The ion name string.
        :return: True if the name is valid, otherwise False.
        """
        try:
            assert isinstance(ion, str)
            # check if ion name is of the form element_number
            assert "_" in ion
            # separate ion by where there is _ character and generate a list of the form [element, number]
            ion = ion.split("_")
            assert len(ion) == 2
            # assert ion[0] is in the periodic table of elements
            assert ion[0] in [
                "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al",
                "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe",
                "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y",
                "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te",
                "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb",
                "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w", "re", "os", "ir", "pt",
                "au", "hg", "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th", "pa",
                "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr"
            ]
            assert ion[1].isdigit()
            return True
        except:
            return False
    
    def line2ID(self, name: str = None, wvl: float = None, list_names: List[Tuple[str, float] | int] = None) -> int:
        """
        Converts a line name and wavelength or a list of names and wavelengths to their corresponding IDs.

        :param name: The name of the line.
        :param wvl: The wavelength of the line.
        :param list_names: A list of names and wavelengths.
        :return: The ID of the line or list of IDs.
        """
        assert (name is not None and wvl is not None) or list_names is not None, "you should provide either the name and wvl or the list of names"
        if list_names is None:
            #check if lines contain the line
            assert self._check_valide_ion_name(name), (f"the name: {name} is not a valid ion name according to the CHIANTI format.")
            slected_lines = self._LINES[(self._LINES["name"]==name)&(self._LINES["wvl"]==wvl)]
            if slected_lines.empty:
                raise ValueError(f"the line with name: {name} and wvl: {wvl} doesn't exist in the catalog")
            elif len(slected_lines)>=2:
                raise ValueError(f"the line with name: {name} and wvl: {wvl} is not unique in the catalog")
            return slected_lines.iloc[0]["ID"]
        else: 
            list_ID = []
            for line in list_names:
                assert self._check_valide_ion_name(line[0]), f"The name: {line} is not a valid ion name according to the CHIANTI format."
                try: 
                    lineID = self.line2ID(name=line[0],wvl=line[1])
                except: 
                    raise AssertionError(f"the line with name: {line[0]} and wvl: {line[1]} doesn't exist in the line catalog, please make sure you picked the right line or add it if it doesn't exist.\nline catalog: {self._LINES}")
                list_ID.append(lineID)
            return list_ID
    
    def ID2line(self, ID: int | List[int]) -> Dict:
        """
        Converts a line ID or list of IDs to their corresponding names and wavelengths.

        :param ID: The line ID or list of line IDs.
        :return: A dictionary of line attributes.
        """
        if not isinstance(ID,Iterable):
            #check if lines contain the line
            slected_lines = self._LINES[self._LINES["ID"]==ID]
            if slected_lines.empty:
                raise ValueError(f"the line with ID: {ID} doesn't exist in the catalog")
            elif len(slected_lines)>=2:
                raise ValueError(f"the line with ID: {ID} is not unique in the catalog")
            return slected_lines.iloc[0].to_dict()
        else:
            list_lines = []
            ID_list = ID
            for ID in ID_list:
                slected_lines = self._LINES[self._LINES["ID"]==ID]
                if slected_lines.empty:
                    raise ValueError(f"the line with ID: {ID} doesn't exist in the catalog")
                elif len(slected_lines)>=2:
                    raise ValueError(f"the line with ID: {ID} is not unique in the catalog")
                list_lines.append(slected_lines.iloc[0].to_dict())
            return list_lines
