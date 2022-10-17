from __future__ import annotations
from math import pi, sqrt
import typing as t

_DEFAULT_MEDIUM = 'air@20'


def _isiterable(obj, exclude=str):
    return hasattr(obj, '__iter__') and not isinstance(obj, exclude)


def freq2wavelen(freq: float, speed: str|float = _DEFAULT_MEDIUM) -> float:
    """
    Wavelength of a frequency based on given sound speed

    calculate the wavelength of a given frequency based on the soundspeed given
    soundspeed can be the value of the soundspeed in m/s or an a string defining
    the medium (passed directly to the function speed_of_sound)

    NB: the result is expressed in **meters**

    Args:
        freq: the frequency of the sound
        speed: the speed of sound, or the medium as a string

    Returns:
        the wavelength
    """
    speed = soundspeed(speed)
    wavelength = speed / freq
    return wavelength


def wavelen2freq(wavelength: float, speed: float|str = _DEFAULT_MEDIUM) -> float:
    """
    The frequency corresponding for the given wavelength through the given medium

    Args:
        wavelength: the wavelength of the sound (in meters)
        speed: either the speed of sound or the medium as a string

    Returns:
        the corresponding freq.
    """
    speed = soundspeed(speed)
    freq = speed / wavelength
    return freq


def distance2delay(distance: float, speed: float|str = _DEFAULT_MEDIUM) -> float:
    """
    The delay in seconds for a sound to travel the given distance

    Calculate the delay in seconds for a sound to travel the given distance
    at the indicated speed of sound.

    Args:
        distance: in METERS
        speed: in m/s, or a medium and/or temperature (will me passed sic to soundspeed)

    Returns:
        the time needed for the sound to travel the given distance
    """
    speed = soundspeed(speed)
    time = distance / speed
    return time


def delay2distance(delay: float, speed: float|str = _DEFAULT_MEDIUM) -> float:
    """
    The distance needed for a sound to arrive with a given delay

    Calculate the distance necessary for a sound to arrive with a
    given `delay` at the indicated `speed` of sound

    Args:
        delay: the time delay
        speed: the speed  of sound or the medium

    Returns:
        the distance between the observer and the sound source
    """
    speed = soundspeed(speed)
    distance = delay * speed
    return distance


def celcius2kelvin(temp: float) -> float:
    """
    Convert celcius to kelvin
    """
    return temp + 273.15


def _speed_of_sound_gases(k: float, R: float, T: float) -> float:
    """
    k : ratio of specific heats
    R : gas constant
    MObjT : absolute temperature

    from http://www.engineeringtoolbox.com/speed-sound-d_519.html
    """
    return (k * R * T) ** 0.5


def _speed_of_sound_hooks_law(E: float, p: float) -> float:
    return (E / p) ** 0.5


_GASES = {
    'air'     : {'k': 1.4,  'R': 286.9},
    'helium'  : {'k': 1.66, 'R': 2077},
    'hydrogen': {'k': 1.41, 'R': 4124},
    'nitrogen': {'k': 1.4,  'R': 296.8, 'formula': 'N2'},
}

_LIQUIDS = {
}

_SOLIDS = {
}


def _medium_to_function(medium):
    # type: (str) -> t.Optional[t.Callable[[float], float]]
    medium = medium.lower()
    if medium in _GASES:
        props = _GASES.get(medium)

        def func(temp):
            return _speed_of_sound_gases(props['k'], props['R'], 
                                         celcius2kelvin(t))
        return func
    return None


SOUNDSPEED_SUPPORTED_MEDIA = (
    set(list(_GASES.keys()) + 
        list(_LIQUIDS.keys()) + 
        list(_SOLIDS.keys()))
)


def _parse_medium(medium: str, temp=20.) -> float:
    if isinstance(medium, str):
        if '@' in medium:
            medium, tempstr = medium.split('@')
            temp = float(tempstr)
    elif isinstance(medium, (tuple, list)):
        medium, temp = medium
    return medium, temp


def soundspeed(medium='air', temp=20) -> float:
    """
    The speed of sound for the given temperature and medium

    SOUNDSPEED_SUPPORTED_MEDIA holds a list of valid media.

    Temperature only makes sense for gases.

    .. note::

        For usability, all these function calls mean the same:

        * ``soundspeed('air', 20)``
        * ``soundspeed('air@20')``

    """
    if not isinstance(medium, str):
        speed = float(medium)
        return speed
    medium, temp = _parse_medium(medium, temp)
    func = _medium_to_function(medium)
    if func is None:
        raise ValueError("Medium not supported (see speed_of_sound_supported_media)")
    return func(temp)


def phaseshift(frequency: float, distance: float, medium='air@20') -> float:
    """
    The phase shift in radians of a signal with a given frequency after distance

    .. note:: to calculate the phase-shift after a given time delay, convert it with

    >>> delay2distance(delay, medium)

    .. code::

        phase_in_radians = wave_length * time_difference = 2pi * freq * time_difference

    via: http://www.sengpielaudio.com/calculator-timedelayphase.htm
    """
    timedelay = distance2delay(distance, medium)
    phase_shift = 2 * pi * frequency * timedelay
    return phase_shift


def string_transverse_propagation_speed(tension: float, density: float) -> float:
    """
    Args:
        tension: tension of the string in Newton
        density: mass of the string per unit length, in kg/m

    Returns:
        the propagation speed

    see: https://en.wikipedia.org/wiki/String_vibration
    """
    return sqrt(tension / density)


def longitudinal_propagation_speed(material):
    v = {
        'aluminium': 5082,
    }.get(material)
    return v


def metalrod_longitudinal_wave_freq(length: float, material='aluminium') -> float:
    v = longitudinal_propagation_speed(material)
    if v is None:
        raise ValueError(f"material not known: {material}")
    wavelength = 2 * length
    freq = v / wavelength
    return freq
