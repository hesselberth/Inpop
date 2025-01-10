### NAME
Inpop - Calculate planet positions, lunar librations and time transformations using the high precision INPOP ephemeris.

### SYNOPSIS
Inpop is a Python package for calculating planetary positions using the very high precision INPOP ephemerides.
INPOP ephemerides are produced and published by IMCCE (https://ftp.imcce.fr/pub/ephem/planets/). Published INPOP files contain Chebyshev coefficients for the position of the solar system bodies as well as the libration angles of the moon.

Because the underlying models are based in general relativity, (most) INPOP files also contain time transformations. Since these depend on the positions of the solar system bodies, INPOP is a 4D ephemeris. INPOP files are available in the TCB or TDB timescales. TCB files contain the transformation TCG-TCB (TCGmTDB) and TDB files contain the transformation TT-TDB (TTmTDB). INPOP .dat files are published in big-endian and little-endian byte order and in both a -100 / +100 year time span and a -1000 / +1000 year time span. Files of the latter type typically do not support the time scale transformations needed for the very high precision applications. Inpop aims to support all INPOP binary (.dat) files.

#### REQUIREMENTS
Inpop depends on Numpy. Numba is not required but will result in a speedup when the file is loaded into memory.

#### PROVIDES
The inpop package provides a single class: Inpop. Besides positions, libration angles and time transformations it also computes the first order derivatives.

### DESCRIPTION
An Inpop constructor is called with a binary INPOP (.dat) file name as its argument:

```python
from inpop import Inpop
inpop = Inpop("inpop21a_TDB_m100_p100_tt.dat")
```

This will open the INPOP file. If the file is not found it will be downloaded automatically to the path given (if the path is writable). It is allowed to omit the filename completely, in this case the most recent 200 year TDB file is opened in the current working directory or, if not found, it is first downloaded to this loation. Small INPOP files (-100 / +100 years) are loaded into memory for fast calculations, accelerated by just-in-time compilation through the LLVM compiler and Numba. If Numba is not available on the system the speedup by holding the file in memory will be significantly smaller.

It is possible to keep the data fully on disk and use seek operations to read the data into buffers:

```python
from inpop import Inpop
inpop = Inpop("inpop21a_TDB_m100_p100_tt.dat", load=False)
```

Similarly, large files (-1000 / +1000 years) can be forced into memory by specifying `load=True`.

Once the Inpop class is initiated we can inspect the time range for which the INPOP file has data:

```python
print(inpop.jd_beg, inpop.jd_end)
2414105.0 2488985.0
```

The time range is given in Julian days in the ephemeris time scale (TDB or TCB). The time scale of this file is TDB:

```python
print(inpop.timescale)
TDB
```

The following important constants are always present:

```python
print(inpop.AU, inpop.EMRAT)
149597870.7 81.30056789872074
```

Besides these constants INPOP files have a constant record which can be accessed through a dictionary:

```python
print(inpop.constants)
{'KSIZER': 1226.0, 'VERSIO': 21.1, 'FVERSI': 0.0, 'FORMAT': 11.0, 'UNITE': 1.0, 'TIMESC': 0.0, 'REDATE': 2021.0624, 'CLIGHT': 299792.458, 'GM_Mer': 4.91248045036476e-11, ...
```

The constant record contains up to 400 values used during the ephemeris integration:

```python
print(len(inpop.constants))
400
```

### The `info()` method
Further details about the internals of the file can be obtained through the `info` member function:

```
INPOP file             inpop21a_TDB_m100_p100_tt.dat
Byte order             Little-endian
Label                  ['INPOP21a', '', '']
JDbeg, JDend, interval 2414105.0, 2488985.0, 32.0
record_size            1226
num_const              400
AU, EMRAT              149597870.7, 81.30056789872074
DENUM                  100
librat_ptr             [819  10   4]
TTmTDB_ptr             [939  12   8]
version                21.1
fversion               0.0
format                 11.0
KSIZER                 1226
UNITE                  1
has_vel                False
has_time               True
has_asteroids          False
unit_pos               au
unit_vel               au/day
unit_time              s
unit_angle             rad
timescale              TDB
```

All of these are instance variables. As can be seen this file has a time conversion from TT to TDB (`has_time` is `True`) and units of position, velocity, time and angle are au, au/day, seconds and radians respectively. This is always the case, even if `UNITE == 0` (meaning file units are km and km/day). Units will be automatically converted to au and au/day.

#### `__str__()`

Above information can be obtained by:

```python
print(inpop)
```

#### `PV(jd, target, center, rate=True, **kwargs)`
This method computes the state of a solar system body (the target) with respect to a center at time jd. jd is the Julian date in the ephemeris time scale (TDB or TCB). Because of the limited precision of a `double` it is advised to split the Julian date in a day part and a time fraction if sub-millisecond accuracy is required. In that case jd should be a `np.array([date, time_fraction], dtype=np.double)` (in that order).

target and center are integers from 0 to 12. The state vectors are returned as a numpy array [P, V] of type np.double. P and V are both 3-vectors containing the position and velocity respectively. The encoding for the target and the center is as follows:

```
 0 mercury
 1 venus
 2 earth
 3 mars
 4 jupiter
 5 saturn
 6 uranus
 7 neptune
 8 pluto
 9 moon
10 sun
11 ssb
12 emb
```

If the above strings are passed to `PV` they will be converted automatically to the integer codes. For example, the position of the moon with respect to the earth on January 1 2000 at 12:00 TDB time was:

```python
print(inpop.PV(2451545.0, 'moon', 'earth'))
[[-0.0019492816590940113 -0.0017828919060276236 -0.0005087136946011068]
 [ 0.0003716704750190104 -0.0003846978294553674 -0.0001740301567948636]]
```

The first row is the position in AU, the second row the velocity in AU/day. The distance between the moon and the earth at that moment was:

```python
import numpy as np
print(np.linalg.norm(inpop.PV(2451545.0, 'moon', 'earth')[0] * inpop.AU))
402448.639909165  # km`
```

If the velocity is not required it can be left out as follows:
```python
print(inpop.PV(2451545.0, 'moon', 'earth', rate=False))
[-0.0019492816590940113 -0.0017828919060276236 -0.0005087136946011068]
```

Because the INPOP file opened above contains TDB data (`self.timescale == "TDB"`), the result is given in the dynamical system. Note that there is a rate difference between TDB and TCB clocks and according to general relativity a scale difference occurs between the positions (and masses) as well. Although the correction is of order 1e-8 this falls well within the numerical and data precision of Inpop.

By passing the keyword argument `ts` it is possible to force the result in a specific timescale:

```python
print(np.linalg.norm(inpop.PV(2451545.0, 'moon', 'earth', rate=False, ts="TCB") * inpop.AU))
402448.2845950923 # km
```

As can be seen, there is a difference of 355.314 meters between these distances. The numerical precision for the earth-moon positions is in the 10 um range, much more precise than the available measurements, which are in the mm range.

Note that the TDB-TCB conversions degrade numerical accuracy somewhat, to the 1e-13 AU range for Pluto at 50 AU distance. For maximum accuracy it is advised to use INPOP ephemeris files for the time scale in which data is required.

#### `LBR(jd, rate=True)`
This method computes the physical libration angles of the moon. jd is again the Julian date in the ephemeris time scale. Note that since the earth and the moon rotate in the ICRF, the z-component winds linear in time with an oscillatory component superimposed. The angles are given in radians:

```python
print(inpop.LBR(2451545.0))
[[-5.4147737920634452e-02  4.2485576628776955e-01  7.1866746406895576e-01]
 [-1.1684365616883440e-04  4.5189513573293875e-05  2.3009987407557086e-01]]
```

For increased time accuracy, again pass jd as `np.array([date, time_fraction], dtype=np.double)`. Like `PV`,`LBR` accepts the keyword argument `ts`.

#### `TTmTDB(tt_jd, rate=False)`
Time difference between the TT (terrestrial time) and TDB (barycentric dynamical time) scales, computed from the Julian date in TT. TT and TDB tick on average at the same rate but over the year there are relativistic drifts of the order of a millisecond. TDB = TT - TTmTDB(tt_jd). Given the small correction and the slow change the function can be used both ways. The correction is given in seconds.

```python
print(inpop.TTmTDB(2451545.0))
9.928893069898122e-05  # seconds
```

The accuracy of this result is better than 100ns for the past and next century.
For high precision, pass jd as a date and time array of length 2 (`np.array([date, time_fraction], dtype=np.double)`).

#### `TCGmTCB(tcg_jd, rate=False)`
Time difference between the TCB (barycentric coordinate time) and TCG (geocentric coordinate time) scales.
Again, for high precision, pass jd as a date/time array of length 2 (`np.array([date, time_fraction], dtype=np.double)`).

#### `close()`
Closes the INPOP file (if it was still open).

#### `__del__()`
Destructor, makes sure there are no dangling file pointers

#### A word about precision
Inpop is fully written in python and uses double precision arithmetic. When running tests against reference data the numerical errors are 2e-14 AU for positions and sub-us level for time over 2 centuries. If sub-ms precision is required, 2 floats should be used for the date. The first one can be used for pseudo-integer date arithmetic and the second one for the day fraction. The library will subtract day offsets from the first jd argument without error and then add the day fraction to a much smaller date, resulting in high precision calculations.

### AUTHOR
Inpop is written by Marcel Hesselberth.

### REPORTING BUGS
       Inpop online help: https://github.com/hesselberth/Inpop/issues

### COPYRIGHT
Marcel Hesselberth.

1. Inpop is released under GPLv3. The text of the license can be found at https://www.gnu.org/licenses/gpl-3.0.txt .
   This is free software: you are free to change and redistribute it. There is NO WARRANTY.

2. The software neither has any relation to IMCCE nor to Leiden University.

3. INPOP data and other files on the IMCCE web server are the intellectual property of the IMCCE. For their conditions of use, see https://www.imcce.fr/mentions-legales.

### SEE ALSO
CIP, CNAV and the documentation in inpop.py.
