# Copyright (c) 2026 Brady Griffith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SourceParams class to manage calibration source energy and features.

Needs:
    - Lookup by name
    - List of gamma lines
    - Block out regions
"""
from typing import Optional, Self
import logging
import numpy as np


class SourceParams:
    """Collection of all lines for a gamma source."""
    _source_dict = dict()

    def __init__(self,
                 energies: np.typing.NDArray[np.float64],
                 name: Optional[str] = None
                 ) -> None:
        """Initialize the source with its energy.

        Args:
            energies:
                Array of source lines in keV
            name:
                Name of source. For example, 'Ba133'. No form is required,
                but the [Element Abbreviation][Isotope] is used for all
                provided sources.
        """
        self.energies = np.array(energies, dtype=np.float64)

        if name is not None:
            if name in self._source_dict:
                logging.warning('Overwriting SourceParam entry for %s', name)
            self._source_dict[name] = self

    def get_filter(self,
                   points: np.typing.NDArray[np.long],
                   gain: float
                   ) -> np.typing.NDArray[np.bool]:
        """Create a boolean array to remove points away from target energies.

        Args:
            points: Array of detector values for a set of events.
            gain: Detector gain in (points units) / keV.

        Returns:
            Boolean array, true for all points near an energy for this source.
        """
        amplitudes = np.sum(points, axis=1)

        energy_floor = gain * min(self.energies) / 2
        energy_ceiling = gain * 2. * max(self.energies)

        return (amplitudes >= energy_floor) & (amplitudes <= energy_ceiling)

    @classmethod
    def get_source(cls, name: str) -> Self:
        """For a named isotope, return its """
        if name in cls._source_dict:
            return cls._source_dict[name]
        else:
            logging.warning("No source named %s.", name)
            return cls([])

# Add sources to the library
# TODO Add sources
