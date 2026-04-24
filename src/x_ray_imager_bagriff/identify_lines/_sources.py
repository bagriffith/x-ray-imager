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

"""SourceParams class to manage calibration source energy and features."""
from typing import Optional, Self
import logging
import numpy as np


def check_gain_range(gain: Optional[float] = None,
                     gain_range: Optional[tuple[float, float]] = None
                     ) -> tuple[float, float]:
    """Validate a gain or gain_range.

    Args:
        gain: The max and min gain (dector response / energy).
        gain_range:

    Returns:
        Tuple of the either `gain_range` or `gain` twice,ordered
        (smallest, largest).

    Raises:
        ValueError: `gain_range` is invalid. Also if too many or too few
            arguments are provided.
    """
    if gain_range is None:
        if gain is None:
            raise ValueError('Must specify either gain or gain_range.')

        gain_range = (gain, gain)
    else:
        if gain is not None:
            raise ValueError('gain and gain_range both provided.')

    if len(gain_range) != 2:
        raise ValueError('Range must be specified as two values.')

    if gain_range[1] < gain_range[0]:
        logging.warning("Gain range was reversed: %s.", gain_range)
        gain_range = (gain_range[1], gain_range[0])

    return gain_range


class SourceParams:
    """Collection of all lines for a gamma source."""
    _source_dict = dict()

    def __init__(self,
                 energies: np.typing.ArrayLike,
                 name: Optional[str] = None
                 ) -> None:
        """Initialize the source with its energy.

        Args:
            energies: Array of source lines in keV
            name: Name of source. For example, 'Ba133'. No form is
                required, but the [Element Abbreviation][Isotope]
                is used for all provided sources.
        """
        self.energies = np.array(energies, dtype=np.float64)
        if len(self.energies) == 0:
            # Empty list is valid, but unlikely to be used outside an error.
            logging.warning("Empty list of energies for SourceParams %s.",
                            name)

        self.name = name
        if name is not None:
            if name in self._source_dict:
                logging.warning('Overwriting SourceParams entry for %s', name)
            self._source_dict[name] = self

    def __len__(self):
        return len(self.energies)

    def get_filter(self,
                   points: np.typing.ArrayLike,
                   gain: Optional[float] = None,
                   gain_range: Optional[tuple[float, float]] = None,
                   bumper: Optional[float] = 1.5
                   ) -> np.typing.NDArray[np.bool]:
        """Create a boolean array to remove points away from target energies.

        Args:
            points: Array of detector values for a set of events.
            gain: Detector gain in (points units) / keV.
            gain_range: Expand the filters to cover this entire gain range.

        Returns:
            Boolean array, true for all points near an energy for this source.

        Raises:
            ValueError: An invalid gain/gain range is provided.
        """
        gain_range = check_gain_range(gain, gain_range)

        if gain_range is None:
            return np.full(np.shape(points)[0], True)

        amplitudes = np.sum(points, axis=1)

        energy_floor = gain_range[0] * min(self.energies) / bumper
        energy_ceiling = gain_range[1] * bumper * max(self.energies)
        logging.info('Energy_range: %f.1 %f.1', energy_floor, energy_ceiling)

        return (amplitudes >= energy_floor) & (amplitudes <= energy_ceiling)

    @classmethod
    def get_source(cls, name: str) -> Self:
        """For a named isotope, return its SourceParams if created.

        Args:
            name: Name of source used at creation.
        """
        if name in cls._source_dict:
            return cls._source_dict[name]
        else:
            logging.warning("No source named %s.", name)
            return cls([])

    @classmethod
    def source_choices(cls):
        return list(cls._source_dict.keys())


# Add sources to the library
SourceParams([15.5, 59.6], 'Am241')
SourceParams([22.5, 88.0], 'Cd109')
SourceParams([123.6], 'Co57')
SourceParams([511.0], 'Na22')
