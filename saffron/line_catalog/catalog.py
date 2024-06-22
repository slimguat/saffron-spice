import json
from typing import Dict, List
import pkg_resources
import numpy as np

import pandas as pd


class LineCatalog:
    def __init__(self, file_location: str = None, verbose=0):
        """
        Initializes the LineCatalog class.

        :param file_location: Path to the JSON file containing the line catalog data.
        :param verbose: showing more info.
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
        with open(self.PATH, "r") as f:
            if self.verbose >= 1:
                print("loading from ", self.PATH)
            self._CATALOGUE = json.load(f)

    def dump(self, new_path: str = "::SAME") -> None:
        """
        Dumps the line catalog and spectral windows catalog in the specified format.

        :param new_path: Path to save the formatted catalog. If set to "::SAME", the current file location will be used.
        """
        if new_path == "::SAME":
            new_path = self.PATH

        dumpable = {
            "LINES": [
                {"name": name, "wvl": wvl}
                for name, wvl in zip(self._LINES["name"], self._LINES["wvl"])
            ],
            "SPECTRAL_WINDOWS_CATALOGUE": [
                {"lines": lines, "max_line": max_line}
                for lines, max_line in zip(
                    self._WINDOWS["lines"], self._WINDOWS["max_line"]
                )
            ],
        }

        with open(new_path, "w") as f:
            f.write("{\n")
            f.write('    "LINES": [\n')
            for line in dumpable["LINES"]:
                # f.write(f"        {json.dumps(line, separators=(',', ':'))}"+(",\n" if line!=dumpable["LINES"][-1] else "\n"))
                name = '"' + line["name"] + '"'
                f.write(
                    f'        {{"name":{name: <10s},"wvl":{line["wvl"]:7.2f}}}'
                    + (",\n" if line != dumpable["LINES"][-1] else "\n")
                )
            f.write("    ],\n")
            f.write('\n    "SPECTRAL_WINDOWS_CATALOGUE": [\n')
            for window in dumpable["SPECTRAL_WINDOWS_CATALOGUE"]:
                # f.write(f"        {json.dumps(window, separators=(',', ':'))}"+(",\n" if window!=dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1] else "\n"))
                # f.write(f"        {json.dumps(window, separators=(',', ':'))}"+(",\n" if window!=dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1] else "\n"))
                lines = str(window["lines"]).replace("'", '"')
                max_line = '"' + window["max_line"] + '"'
                f.write(
                    f'        {{"lines":{lines: <40s},"max_line":{max_line: <10s}}}'
                    + (
                        ",\n"
                        if window != dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1]
                        else "\n"
                    )
                )

            f.write("    ]\n")
            f.write("}\n")

    def _update_catalog(self) -> None:
        """
        Updates the line catalog data.
        """
        self._CATALOGUE["LINES"] = self._LINES.to_dict()
        self._CATALOGUE["SPECTRAL_WINDOWS_CATALOGUE"] = self._WINDOWS.to_dict()

    def add_window(self, lines: List[str], max_line: str) -> None:
        """
        Adds a new spectral window to the line catalog.

        :param lines: List of lines in the new window.
        :param max_line: Maximum line in the new window.
        """
        found = self._WINDOWS.lines.apply(lambda x: set(x) == set(lines))
        if found.any():
            raise ValueError(
                f"the window is already there if you want to modify the max delete the window first"
            )
        if max_line not in lines:
            raise ValueError(f"max_line:{max_line} should be part of the lines:{lines}")

        for line in lines:
            if not self._LINES.name.str.contains(line).any():
                raise ValueError(
                    f"the line: {line} doesn't exist in the catalog add it first before adding a windows that contains this line\n use LineCatalog.new_line() to do it"
                )

        lines.sort()
        dflines = pd.DataFrame({"lines": [lines], "max_line": max_line}, index=[0])

        self._WINDOWS = pd.concat([self._WINDOWS, dflines], axis=0, ignore_index=True)
        self._update_catalog()

    def add_line(self, name: str, wvl: float) -> None:
        """
        Adds a new line to the line catalog.

        :param name: Name of the new line.
        :param wvl: Wavelength of the new line.
        """
        if self._LINES.name.str.contains(name).any():
            raise ValueError(
                f"the line:{name} does exist.\nIf it's another line try to add a number at the end or increase it.\nIf you want to replace it delete it first than replace it"
            )
        if self.verbose >= 1:
            print(
                "you are adding a new spectral line it would be great if you use CHIANTI structure element_number-in-roman_number-if-there-are-multiple"
            )
        self._LINES = pd.concat(
            [
                self._LINES,
                pd.DataFrame({"name": name, "wvl": wvl}, index=[0]),
            ],
            axis=0,
            ignore_index=True,
        )
        self._LINES = self._LINES.sort_values(by="wvl")
        self._update_catalog()

    def remove_window(self, lines: List[str]) -> None:
        """
        Removes a spectral window from the line catalog.

        :param lines: List of lines in the window to be removed.
        """
        lines.sort()
        found = self._WINDOWS.lines.apply(lambda x: set(x) == set(lines))
        if not found.any():
            raise ValueError(
                f"the window to remove with this composition: {lines} doesn't exist "
            )
        else:
            self._WINDOWS = self._WINDOWS[np.logical_not(found)]
            self._update_catalog()

    def remove_line(self, name: str) -> None:
        """
        Removes a line from the line catalog.

        :param name: Name of the line to be removed.
        """
        found = self._LINES.name.apply(lambda x: x == name)
        windows_with_name_in = self._WINDOWS.lines.apply(lambda x: name in x)
        if not found.any():
            raise ValueError(
                f"The line to remove with this name: {name} doesn't exist "
            )
        elif windows_with_name_in.any():
            raise ValueError(
                f"The line:{name} is in some windows\n{self._WINDOWS[windows_with_name_in]}\n you should delete them first"
            )
        else:
            self._LINES = self._LINES[np.logical_not(found)]
            self._update_catalog()

    # getters
    def get_catalog(self) -> Dict:
        return self._CATALOGUE

    def get_catalog_lines(self) -> pd.DataFrame:
        return self._LINES

    def get_catalog_windows(self) -> pd.DataFrame:
        return self._WINDOWS

    def get_line_wvl(self, lines):
        if isinstance(lines, str):
            # print(self._LINES.loc[self._LINES["name"]==lines,'wvl'],type(self._LINES.loc[self._LINES["name"]==lines,'wvl']))
            wvl = self._LINES.loc[self._LINES["name"] == lines, "wvl"]
            if wvl.empty:
                return None
            return wvl.iloc[0]
        else:
            return np.array([self.get_line_wvl(i) for i in lines])


# TODO DELETE
# import json
# from typing import Dict, List
# import os
# import pandas as pd
# from pathlib import Path
# class LineCatalogue:
#     def __init__(self, file_location: str = "::cat_file_loc/SPICE_SpecLines.json"):
#         """
#         Initializes the LineCatalogue class.

#         :param file_location: Path to the JSON file containing the line catalogue data.
#         """
#         self.PATH = file_location
#         if ("::cat_file_loc/"in file_location):
#             script_dir = os.path.dirname(os.path.abspath(__file__))
#             _file_path = self.PATH[file_location.find("::cat_file_loc/")+len('::cat_file_loc/'):]
#             self.PATH = (Path(script_dir)/_file_path)
#         self.verbose = 1
#         self.load()
#         self._LINES = pd.DataFrame(self._CATALOGUE["LINES"])
#         self._WINDOWS = pd.DataFrame(self._CATALOGUE["SPECTRAL_WINDOWS_CATALOGUE"])

#     def load(self) -> None:
#         """
#         Loads the line catalogue data from the JSON file.
#         """

#         with open(self.PATH, "r") as f:
#             self._CATALOGUE = json.load(f)

#     def dump(self, new_path: str = "::SAME") -> None:
#         """
#         Dumps the line catalogue and spectral windows catalogue in the specified format.

#         :param new_path: Path to save the formatted catalogue. If set to "::SAME", the current file location will be used.
#         """
#         if new_path == "::SAME":
#             new_path = self.PATH

#         dumpable = {
#             "LINES": [{"name": name, "wvl": wvl} for name, wvl in zip(self._LINES["name"], self._LINES["wvl"])],
#             "SPECTRAL_WINDOWS_CATALOGUE": [
#                 {"lines": lines, "max_line": max_line} for lines, max_line in zip(self._WINDOWS["lines"], self._WINDOWS["max_line"])
#             ],
#         }

#         with open(new_path, "w") as f:
#             f.write("{\n")
#             f.write('    "LINES": [\n')
#             for line in dumpable["LINES"]:
#                 # f.write(f"        {json.dumps(line, separators=(',', ':'))}"+(",\n" if line!=dumpable["LINES"][-1] else "\n"))
#                 name = "\""+line["name"]+"\""
#                 f.write(f'        {{"name":{name: <10s},"wvl":{line["wvl"]:7.2f}}}'+(",\n" if line!=dumpable["LINES"][-1] else "\n"))
#             f.write("    ],\n")
#             f.write('\n    "SPECTRAL_WINDOWS_CATALOGUE": [\n')
#             for window in dumpable["SPECTRAL_WINDOWS_CATALOGUE"]:
#                 # f.write(f"        {json.dumps(window, separators=(',', ':'))}"+(",\n" if window!=dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1] else "\n"))
#                 # f.write(f"        {json.dumps(window, separators=(',', ':'))}"+(",\n" if window!=dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1] else "\n"))
#                 lines = str(window["lines"]).replace('\'','\"')
#                 max_line = "\""+window["max_line"]+"\""
#                 f.write(f'        {{"lines":{lines: <40s},"max_line":{max_line: <10s}}}'+(",\n" if window!=dumpable["SPECTRAL_WINDOWS_CATALOGUE"][-1] else "\n"))

#             f.write("    ]\n")
#             f.write("}\n")

#     def _update_catalogue(self) -> None:
#         """
#         Updates the line catalogue data.
#         """
#         self._CATALOGUE["LINES"] = self._LINES.to_dict()
#         self._CATALOGUE["SPECTRAL_WINDOWS_CATALOGUE"] = self._WINDOWS.to_dict()

#     def add_window(self, lines: List[str], max_line: str) -> None:
#         """
#         Adds a new spectral window to the line catalogue.

#         :param lines: List of lines in the new window.
#         :param max_line: Maximum line in the new window.
#         """
#         found = self._WINDOWS.lines.apply(lambda x: set(x) == set(lines))
#         if found.any():
#             raise ValueError(f"the window is already there if you want to modify the max delete the window first")
#         if max_line not in lines:
#             raise ValueError(f"max_line:{max_line} should be part of the lines:{lines}")

#         for line in lines:
#             if not self._LINES.name.str.contains(line).any():
#                 raise ValueError(
#                     f"the line: {line} doesn't exist in the catalogue add it first before adding a windows that contains this line\n use LineCatalogue.new_line() to do it"
#                 )

#         lines.sort()
#         dflines = pd.DataFrame({"lines": [lines], "max_line": max_line}, index=[0])

#         self._WINDOWS = pd.concat([self._WINDOWS, dflines], axis=0, ignore_index=True)
#         self._update_catalogue()

#     def add_line(self, name: str, wvl: float) -> None:
#         """
#         Adds a new line to the line catalogue.

#         :param name: Name of the new line.
#         :param wvl: Wavelength of the new line.
#         """
#         if self._LINES.name.str.contains(name).any():
#             raise ValueError(
#                 f"the line:{name} does exist.\nIf it's another line try to add a number at the end or increase it.\nIf you want to replace it delete it first than replace it"
#             )
#         if self.verbose >= 1:
#             print(
#                 "you are adding a new spectral line it would be great if you use CHIANTI structure element_number-in-roman_number-if-there-are-multiple"
#             )
#         self._LINES = pd.concat(
#             [
#                 self._LINES,
#                 pd.DataFrame({"name": name, "wvl": wvl}, index=[0]),
#             ],
#             axis=0,
#             ignore_index=True,
#         )
#         self._LINES = self._LINES.sort_values(by="wvl")
#         self._update_catalogue()

#     def remove_window(self, lines: List[str]) -> None:
#         """
#         Removes a spectral window from the line catalogue.

#         :param lines: List of lines in the window to be removed.
#         """
#         lines.sort()
#         found = self._WINDOWS.lines.apply(lambda x: set(x) == set(lines))
#         if not found.any():
#             raise ValueError(f"the window to remove with this composition: {lines} doesn't exist ")
#         else:
#             self._WINDOWS = self._WINDOWS[np.logical_not(found)]
#             self._update_catalogue()

#     def remove_line(self, name: str) -> None:
#         """
#         Removes a line from the line catalogue.

#         :param name: Name of the line to be removed.
#         """
#         found = self._LINES.name.apply(lambda x: x == name)
#         windows_with_name_in = self._WINDOWS.lines.apply(lambda x: name in x)
#         if not found.any():
#             raise ValueError(f"The line to remove with this name: {name} doesn't exist ")
#         elif windows_with_name_in.any():
#             raise ValueError(
#                 f"The line:{name} is in some windows\n{self._WINDOWS[windows_with_name_in]}\n you should delete them first"
#             )
#         else:
#             self._LINES = self._LINES[np.logical_not(found)]
#             self._update_catalogue()

#     # getters
#     def get_catalogue(self) -> Dict:
#         return self._CATALOGUE

#     def get_catalogue_lines(self) -> pd.DataFrame:
#         return self._LINES

#     def get_catalogue_windows(self) -> pd.DataFrame:
#         return self._WINDOWS
