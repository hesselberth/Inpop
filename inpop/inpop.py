#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact library for using INPOP files.

Created on Fri Dec  4 14:16:35 2024

@author: Marcel Hesselberth

Version: 0.5
"""

from .constants import Lb, LKb, T0, TDB0_jd
from .cnumba import cnjit
import numpy as np
import struct
from os import path, stat, unlink, SEEK_END
from sys import byteorder
from configparser import ConfigParser


CONFIGFILE = "inpop.ini"  # In the installation directory
FILE_TRESHOLD = 50e6         # Bytes


lpath = path.realpath(path.dirname(__file__))  # The path to this library
config = ConfigParser()
# The config file of this library
config.read(path.join(lpath, CONFIGFILE))


@cnjit(signature_or_function='UniTuple(float64[:], 2)(float64, int64)')
def chpoly(x, degree):
    """
    Evaluate the Chebyshev polynomial and its derivatives at x.

    Best algorithm according to https://arxiv.org/abs/1312.5677v2 .

    Parameters
    ----------
    x      : float.
             Domain [-1, 1].
    degree : int.
             Degree of the polynomial.

    Returns
    -------
    Polynomials T and (derivative) D.
    Both are arrays of length <degree>.

    """
    T = np.ones(degree, dtype=np.double)
    D = np.zeros(degree, dtype=np.double)
    T[1] = x
    D[1] = 1
    for i in range(2, degree):
        T[i] = 2.0 * x * T[i-1] - T[i-2]
        D[i] = 2.0 * T[i-1] + 2.0 * x * D[i-1] - D[i-2]
    return T, D


@cnjit(signature_or_function='f8[:,:](f8, f8, i4, i4, i4, f8[:], f8, f8, i4, i4)')
def calcm(jd, jd2, offset, ncoeffs, ngranules, data,
          jd_beg, interval, nrecords, recordsize):
    """
    Compute a state vector (3-vector and its derivative) from data in memory.

    This is the INPOP decoding routine common to all calculations, whether
    6d (position-velocity), 3d (libration angles) or 1d (time).
    calcm is an accelerated version of Inpop.calc.

    Parameters
    ----------
    jd         : float
                 Julian date
    jd2        : float
                 Julian date (time fraction)
    offset     : int
                 coeff_ptr[0]
    ncoeffs    : int
                 coeff_ptr[1]
    ngranules  : int
                 coeff_ptr[2]
    data       : ndarray
                 self.data
    jd_beg     : float
                 self.jd_beg
    interval   : float
                 self.interval
    nrecords   : int
                 self.nrecords
    recordsize : int
                 self.recordsize

    Returns
    -------
    ndarray
        2x3 matrix pos, vel

    """
    record = int(((jd - jd_beg) + jd2) // interval) + 1
    if record < nrecords:
        record += 1
    raddr = record * recordsize
    jdl = data[raddr]
    span = interval / ngranules
    granule = int(((jd - jdl) + jd2) // span)
    jd0 = jdl + granule * span
    tc = 2 * (((jd-jd0) + jd2) / span) - 1
    gaddr = int(raddr + (offset - 1 + 3 * granule * ncoeffs))
    cx = np.copy(data[gaddr: gaddr + ncoeffs])
    cy = np.copy(data[gaddr + ncoeffs: gaddr + 2 * ncoeffs])
    cz = np.copy(data[gaddr + 2 * ncoeffs: gaddr + 3 * ncoeffs])
    T, D = chpoly(tc, ncoeffs)
    T = np.ascontiguousarray(T)
    D = np.ascontiguousarray(D)
    px = np.dot(cx, T)
    py = np.dot(cy, T)
    pz = np.dot(cz, T)
    vx = np.dot(cx, D) * ngranules
    vy = np.dot(cy, D) * ngranules
    vz = np.dot(cz, D) * ngranules
    return np.array([[px, py, pz], [vx, vy, vz]], dtype=np.double)


class Inpop:
    """Decode Inpop .dat files and compute planetary positions and moon librations."""

    # These differ from NAIF codes.
    bodycodes = {"mercury": 0, "venus": 1, "earth": 2, "mars": 3, "jupiter": 4,
                 "saturn": 5, "uranus": 6, "neptune": 7, "pluto": 8, "moon": 9,
                 "sun": 10, "ssb": 11, "emb": 12}

    def __init__(self, filename=None, load=None):
        """
        Inpop constructor.

        Class to compute state vectors from the 4d INPOP ephemeris.
        Data is read from the .dat file using the INPOP binary file format.
        The file may have little or big endian byte order and TDB or TCB
        time scales.

        If no filename is given, a default file 200 year TDB file is used.
        If the file is not found, it is downloaded to the ephem directory.
        Download will only be attempted if no directory is given. Files only
        download to the default directory to reduce server traffic.

        Parameters
        ----------
        filename : string
                   Path to an INPOP .dat file
        load :     bool, optional
                   If True, the file is completely loaded into memory.
                   If false, the file is accessed fully through seek operations.
                   The default is None, which loads the file in memory if the
                   file size is below FILE_TRESHOLD (about 50MB).

        Returns
        -------
        None.

        """
        self.file = None

        if not filename:
            filename = config["inpopfile"]["default"]
        ext = filename.rsplit(".", 1)[-1]
        if not ext == config["inpopfile"]["ext"]:
            raise (IOError("File extension must be .dat"))

        if byteorder == "little":
            self.machine_byteorder = "<"
            self.opposite_byteorder = ">"
        else:
            self.machine_byteorder = ">"
            self.opposite_byteorder = "<"
        self.byteorder = self.machine_byteorder

        if not path.isfile(filename):
            self.path = self.search(filename)
        else:
            self.path = filename

        try:
            size = stat(self.path).st_size
        except:
            size = 0

        if not size:
            raise (FileNotFoundError(filename))

        if not isinstance(load, bool):
            if size > FILE_TRESHOLD:
                load = False
            else:
                load = True

        self.mem = load
        self.open()

    def search(self, filename):
        """
        Look for the INPOP file and attempt download if file not found.

        Files are always downloaded to the specified directory or to the
        CWD if no directory is given.

        Parameters
        ----------
        filename : string
                   filename of an INPOP .dat file

        Returns
        -------
        path : string
               path to the inpop file.
        """
        dirname, inpopfilename = path.split(filename)
        if dirname == "":
            dirname = "."
        if path.isdir(dirname):
            inpop_version = inpopfilename.split("_", 1)[0]
            if inpop_version[:5] != "inpop":
                raise (IOError(f"Bad filename: {filename}"))
            else:
                pass  # download file
        else:
            raise(FileNotFoundError("Path to INPOP file does not exist."))
        # try download
        try:
            testpath = path.join(dirname, "test")
            file = open(testpath, "w")
            file.write("test")
            file.close()
            unlink(testpath)
        except:
            raise (IOError(f"directory is not writable ({dirname})"))
        url = config["ftp"]["base_url"] + inpop_version + "/" + inpopfilename
        print(f"Downloading {url} to {filename} ...")
        import urllib.request
        try:
            urllib.request.urlretrieve(url, filename)
        except:
            pass  # results in FileNotFoundError
        return filename

    def open(self):
        """
        Open the existing INPOP file.

        Read the header information, the constant values and initialize the
        lookup of Chebyshev polynomials. Some important variables are always
        present (AU, EMRAT, DENUM) and become member variables. According to
        specification 2.0 INPOP files can also contain asteroid information.
        Such files are not in the public domain, hence retrieving asteroid
        information is not (yet) implemented.

        Returns
        -------
        None.

        """
        self.file = open(self.path, 'rb')  # INPOP files are binary

        # Decode the header record
        header_spec = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
        header_struct = struct.Struct(header_spec)
        bytestr = self.file.read(header_struct.size)
        hb = header_struct.unpack(bytestr)  # header block
        self.DENUM = hb[44]  # Must be 100 for INPOP
        if self.DENUM != 100:
            self.file.seek(0)
            self.byteorder = self.opposite_byteorder
            header_spec = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
            header_struct = struct.Struct(header_spec)
            bytestr = self.file.read(header_struct.size)
            hb = header_struct.unpack(bytestr)  # header block
            self.DENUM = hb[44]
            if self.DENUM != 100:
                raise (IOError("Can't determine INPOP file byteorder."))

        self.jd_struct = struct.Struct(f"{self.byteorder}dd")  # julian dates

        self.label = []  # Ephemeris label, list of 3 strings
        self.label.append(hb[0][:84].decode().strip())
        self.label.append(hb[0][84:168].decode().strip())
        self.label.append(hb[0][168:].decode().strip())

        const_names = [hb[1][6*i:6*(i+1)] for i in range(400)]

        self.jd_beg = hb[2]          # Julian start date
        self.jd_end = hb[3]          # Julian end date
        self.interval = hb[4]        # Julian interval
        self.num_const = hb[5]       # Number of constants in the second record
        self.AU = hb[6]              # Astronomical unit
        self.EMRAT = hb[7]           # Mearth / Mmoon
        self.coeff_ptr = [(hb[8+3*i:8+3*i+3]) for i in range(12)]
        self.DENUM = hb[44]          # Ephemeris ID
        self.librat_ptr = hb[45:48]  # Libration pointer
        self.recordsize = hb[48]     # Size of the record in bytes
        self.TTmTDB_ptr = hb[49:52]  # Time transformation TTmTDB or TCGmTCB

        # Location, number of coefficients and number of granules
        # for the 12 bodies.
        self.coeff_ptr = np.array(self.coeff_ptr, dtype=int)

        # Location, number of coefficients and number of granules
        # for the libration angles of the moon.
        self.librat_ptr = np.array(self.librat_ptr, dtype=int)

        # Location, number of coefficients and number of granules
        # for the mapping of  TT-TDB or TCG-TCB
        self.TTmTDB_ptr = np.array(self.TTmTDB_ptr, dtype=int)

        # Decode the constant record
        self.file.seek(self.recordsize*8)
        const_struct = struct.Struct(f"{self.byteorder}%id" % (self.num_const))
        bytestr = self.file.read(const_struct.size)
        cb = const_struct.unpack(bytestr)
        const_names = const_names[:self.num_const]
        self.constants = {const_names[i].strip().decode(): cb[i]
                          for i in range(self.num_const)}

        self.version  = self.constants["VERSIO"]       # ephemerus version number
        self.fversion = self.constants["FVERSI"]       # file version number (0)
        self.format   = self.constants["FORMAT"]       # details about file contents
        self.ksizer   = int(self.constants["KSIZER"])  # numbers per record

        # Decode file format
        self.has_vel = (self.format//1 % 10) == 0
        self.has_time = (self.format//10) % 10 == 1
        self.has_asteroids = (self.format//100) % 10 == 1

        # Use the following unit base and transform where necessary
        self.unit_time  = "s"
        self.unit_angle = "rad"
        self.unit_pos   = "au"
        self.unit_vel   = "au/day"

        self.rate_factor = 2.0 / self.interval  # chain rule
        if self.constants["UNITE"] == 0:
            self.unite = 0
            self.unit_factor = 1.0
        else:
            self.unite = 1
            self.unit_factor = 1.0 / self.AU

        # If no timescale is found it is TDB (file version 1.0)
        if "TIMESC" in self.constants:
            if self.constants["TIMESC"] == 0:
                self.timescale = "TDB"
            else:
                self.timescale = "TCB"
        else:
            self.timescale = "TDB"

        self.nrecords = int((self.jd_end - self.jd_beg) / self.interval)

        if self.mem:
            self.load()
            self.file.close()
            self.file = None

        self.earthfactor = -1 / (1 + self.EMRAT)
        self.moonfactor = self.EMRAT / (1 + self.EMRAT)

    def load(self):
        """
        Load the INPOP file entirely into memory.

        This option speeds up the calculations by avoiding file operations.
        This option also allows Numba acceleration.

        Returns
        -------
        None.

        """
        self.file.seek(0, SEEK_END)
        size = self.file.tell()
        self.file.seek(0)
        if size % 8 != 0:
            raise (IOError("INPOP File has wrong length."))
        data = np.frombuffer(self.file.read(size), dtype=np.double)
        if self.byteorder != self.machine_byteorder:
            data = data.byteswap()  # Changes data (newbyteorder changes view)
        self.data = np.copy(data)   # Changes array status for Numba

    def info(self):
        """
        Generate a string containing information about the INPOP file.

        Returns
        -------
        s : string

        """
        if self.byteorder == '>':
            b = "Big-endian"
        else:
            b = "Little-endian"
        s = f"INPOP file             {self.path}\n"
        s += f"Byte order             {b}\n"
        s += f"Label                  {self.label}\n"
        s += f"JDbeg, JDend, interval {self.jd_beg}, {self.jd_end}, "
        s += f"{self.interval}\n"
        s += f"record_size            {self.recordsize}\n"
        s += f"num_const              {self.num_const}\n"
        s += f"AU, EMRAT              {self.AU}, {self.EMRAT}\n"
        s += f"DENUM                  {self.DENUM}\n"
        s += f"librat_ptr             {self.librat_ptr}\n"
        s += f"TTmTDB_ptr             {self.TTmTDB_ptr}\n"
        s += f"version                {self.version}\n"
        s += f"fversion               {self.fversion}\n"
        s += f"format                 {self.format}\n"
        s += f"KSIZER                 {self.ksizer}\n"
        s += f"UNITE                  {self.unite}\n"

        s += f"has_vel                {self.has_vel}\n"
        s += f"has_time               {self.has_time}\n"
        s += f"has_asteroids          {self.has_asteroids}\n"

        s += f"unit_pos               {self.unit_pos}\n"
        s += f"unit_vel               {self.unit_vel}\n"
        s += f"unit_time              {self.unit_time}\n"
        s += f"unit_angle             {self.unit_angle}\n"
        s += f"timescale              {self.timescale}"
        # s += f"\ncoeff_ptr:\n{self.coeff_ptr}"
        return s

    def __str__(self):
        """
        Enable printing of an Inpop instance.

        Returns
        -------
        string info()

        """
        return self.info()

    def calc(self, jd1, jd2, coeff_ptr, rate):
        """
        Calculate a state vector for a single body.

        This is the Inpop decoding routine common to the calculations, whether
        6d (position-velocity), 3d (libration angles) or 1d (time).
        The file record is located and checked and subsequently the INPOP
        granule with the coefficients is seeked. Based on the granule size
        the Chebyshev time tc is calculated. If the file is loaded in memory,
        numba-accelerated calcm is called.

        Parameters
        ----------
        jd        : float
                    Date in ephemeris time.
        coeff_ptr : ndarray(dtype=int)
                    offset, ncoeffs, ngranules
                    record offset, number of Chebyshev coefficients number of
                    granules
        rate:       bool
                    Whether to return only the value (position, angle, time)
                    or to include its derivative. The default depends on the
                    caller. This option changes the dimension of the result.

        Returns
        -------
        2x3 matrix pos, vel or (if rate=False) 3-vector pos.

        """
        jd = jd1 + jd2
        if jd < self.jd_beg or jd > self.jd_end:
            raise (ValueError("Julian date must be between %.1f and %.1f."
                              % (self.jd_beg, self.jd_end)))
        offset, ncoeffs, ngranules = coeff_ptr
        if self.mem:
            pos, vel = calcm(jd1, jd2, offset, ncoeffs, ngranules,
                             self.data, self.jd_beg, self.interval,
                             self.nrecords, self.recordsize)
            if rate:
                return np.array([pos, vel*self.rate_factor], dtype=np.double)
            return pos
        else:  # file based
            if not self.file:
                raise (IOError(f"Ephemeris file ({self.path}) not open."))
        record = int(((jd1 - self.jd_beg) + jd2) // self.interval) + 1
        if record < self.nrecords:
            record += 1
        raddr = record * self.recordsize * 8  # locate record
        self.file.seek(raddr)
        bytestr = self.file.read(self.jd_struct.size)  # read record limits
        jdl, jdh = self.jd_struct.unpack(bytestr)
        assert (jd >= jdl and jd <= jdh)  # check
        span = self.interval / ngranules
        granule = int(((jd1 - jdl) + jd2) // span)  # compute the granule
        jd0 = jdl + granule * span
        tc = 2 * (((jd1-jd0) + jd2) / span) - 1  # Chebyshev time in granule
        assert (tc >= -1 and tc <= 1)
        gaddr = int(raddr+(offset - 1 + 3 * granule * ncoeffs) * 8)  # -1 for C
        self.file.seek(gaddr)  # read 3 * ncoeffs 8 bit doubles
        coeffs = np.frombuffer(self.file.read(24 * ncoeffs), dtype=np.double)
        coeffs = coeffs.view(coeffs.dtype.newbyteorder(self.byteorder))
        coeffs.resize((3, ncoeffs))  # 3 x ncoeffs matrix
        T, D = chpoly(tc, ncoeffs)  # 2 x ncoeffs
        pos = np.dot(coeffs, T)
        if rate:
            #print("bar")
            vel = np.dot(coeffs, D) * ngranules * self.rate_factor
            return np.array([pos, vel], dtype=np.double)
        #print("foo")
        return pos

    def jd_unpack(jd):
        """
        Decode Julian date.

        In the methods below, a Julian date may be:
            - a single (64 bit floating point) number
            - an ndarray of length 1
            - an ndarray of length 2
        Internally 2 64 bit floats are used, one for the integer (+ 0.5) date
        and one for the time fraction. The first array value will have a
        record and granule date subtracted, resulting in zero numerical error.
        The second float now only has to be added to the difference, which is
        at most a granule interval in size. This method of coding
        [jdate, jtime] in an ndarray(dtype=np.double) is required when
        sub-millisecond timing is needed. A single float has an error of 10's
        of microseconds, 2 floats used as described above have ns precision.


        Parameters
        ----------
        jd : int, np.double, np.ndarray(dtype=np.double)
             Julian date as either a (floating point) number, an array of
             length 1 or an array of length 2 encoding [jdate, jtime]
             where jtime < 1.

        Returns
        -------
        jd1 : float
        jd2 : float

        """
        if isinstance(jd, np.ndarray):
            if len(jd) == 1:
                jd2 = 0
                jd = jd[0]
            elif len(jd) == 2:
                jd2 = jd[1]
                jd = jd[0]
            else:
                raise (ValueError("JD Array must have length 1 or 2"))
        else:
            jd2 = 0
        return jd, jd2

    def PV(self, jd, t, c, rate=True, **kwargs):
        """
        Position and velocity of a target t relative to center c in the ICRF.

        Positions and velocities are computed using the Chebyshev polynomials
        and their derivatives. The position is given in AU, the velocity
        in AU/day.

        Parameters
        ----------
        jd :      int, np.double, np.ndarray(dtype=np.double)
                  Julian date in ephemeris time. INPOP is distributed in
                  TDB and TCB. timescales (see self.timescale).
        t, c :    integer between 0 and 12
                  Target body and the Center from which it is observed.
        rate :    bool
                  whether to return velocity as well. Default is yes.
        **kwargs: ts: string, "TCB" or "TDB". Forces timescale independent of
                  ephemeris timescale. The library will do the conversions.
                  rate: bool. Determine whether the derivative (velocity) is
                  returned. Default: True. If False, result is a 3-vector.

        0:  Mercury
        1:  Venus 
        2:  Earth
        3:  Mars
        4:  Jupiter
        5:  Saturn
        6:  Uranus
        7:  Neptune
        8:  Pluto 
        9:  Moon
        10: Sun
        11: SSB
        12: EMB

        Returns
        -------
        2x3 matrix [P, V] (or 3-vector P, when rate = False)
        Error upon failure (no ephemeris file found, time outside ephemeris,
        body code invalid.

        """
        # Unpack jd
        jd1, jd2 = Inpop.jd_unpack(jd)

        # Convert target to integer
        if not isinstance(t, (int, np.integer)):
            try:
                t = Inpop.bodycodes[t.lower()]
            except:
                print(f"Unknown target ({t}).\nValid targets are:")
                inverse = {Inpop.bodycodes[x]
                    : x for x in Inpop.bodycodes.keys()}
                for i in range(13):
                    print(f"{i:2d} {inverse[i]}")
                raise (KeyError) from None  # pep-0409

        # Convert center to integer
        if not isinstance(c, (int, np.integer)):
            try:
                c = Inpop.bodycodes[c.lower()]
            except:
                print(f"Unknown center ({c}).\nValid centers are:")
                inverse = {Inpop.bodycodes[x]
                    : x for x in Inpop.bodycodes.keys()}
                for i in range(13):
                    print(f"{i:2d} {inverse[i]}")
                raise (KeyError) from None  # pep-0409

        if t < 0 or t > 12 or c < 0 or c > 12:
            raise (LookupError("Code must be between 0 and 12."))

        # Decode ts argument. Compute relativistic conversions if needed.
        if kwargs and "ts" in kwargs:
            ts = kwargs["ts"]
            timescale = ts.upper()
            if timescale == self.timescale:
                gr_pos_factor = 1
            elif timescale == "TCB" and self.timescale == "TDB":
                TDBmTCB = -Lb * ((jd - T0) + jd2) + TDB0_jd
                jd2 += TDBmTCB
                gr_pos_factor = 1 / (1 - Lb)
            elif timescale == "TDB" and self.timescale == "TCB":
                TCBmTDB = LKb * ((jd - T0) + jd2) - TDB0_jd  # / Kb
                jd2 += TCBmTDB
                gr_pos_factor = 1 / (1 + LKb)  # Kb
            else:
                raise (ValueError("Invalid timescale, must be TDB or TCB."))
        else:
            gr_pos_factor = 1

        # Decode and compute special cases for earth, moon, emb, ssb
        if t == c:
            return np.zeros(6).reshape((2, 3))

        match t:
            case 2:
                target = self.calc(jd1, jd2, self.coeff_ptr[9], rate) \
                    * self.earthfactor \
                    + self.calc(jd1, jd2, self.coeff_ptr[2], rate)
            case 9:
                target = self.calc(jd1, jd2, self.coeff_ptr[9], rate) \
                    * self.moonfactor \
                    + self.calc(jd1, jd2, self.coeff_ptr[2], rate)
            case 11:
                target = np.zeros(6).reshape((2, 3))
            case 12:
                target = self.calc(jd1, jd2, self.coeff_ptr[2], rate)
            case _:
                target = self.calc(jd1, jd2, self.coeff_ptr[t], rate)

        match c:
            case 2:
                center = self.calc(jd1, jd2, self.coeff_ptr[9], rate) \
                    * self.earthfactor \
                    + self.calc(jd1, jd2, self.coeff_ptr[2], rate)
            case 9:
                center = self.calc(jd1, jd2, self.coeff_ptr[9], rate) \
                    * self.moonfactor \
                    + self.calc(jd1, jd2, self.coeff_ptr[2], rate)
            case 11:
                center = np.zeros(6).reshape((2, 3))
            case 12:
                center = self.calc(jd1, jd2, self.coeff_ptr[2], rate)
            case _:
                center = self.calc(jd1, jd2, self.coeff_ptr[c], rate)

        # Relativistic and unit conversions
        result = target - center
        result[0] *= gr_pos_factor * self.unit_factor
        result[1, ...] *= self.unit_factor
        return result

    def LBR(self, jd, rate=True, **kwargs):
        """
        Physical libration angles of the moon.

        Parameters
        ----------
        jd :      np.double (or float)
                  Julian time in ephemeris time. INPOP is distributed in TDB
                  and TCB timescales (see self.timescale).
        rate :    bool
                  whether to return (angular) velocity as well. Default is yes.
        **kwargs: ts: string, "TCB" or "TDB". Forces timescale independent of
                  ephemeris timescale. The library will do the conversions.
                  rate: bool. Determine whether the derivative (velocity) is
                  returned. Default: True. If False, result is a 3-vector.

        Returns
        -------
        np.ndarray(dype="float")
             The 3 physical libration angles in radians and the angular
             velocities. (When rate = False, only the angles)

        """
        jd1, jd2 = Inpop.jd_unpack(jd)

        if kwargs and "ts" in kwargs:
            ts = kwargs["ts"]
            timescale = ts.upper()
            if timescale == self.timescale:
                pass
            elif timescale == "TCB" and self.timescale == "TDB":
                TDBmTCB = -Lb * ((jd - T0) + jd2) + TDB0_jd
                jd2 += TDBmTCB
            elif timescale == "TDB" and self.timescale == "TCB":
                TCBmTDB = LKb * ((jd - T0) + jd2) - TDB0_jd  # / Kb
                jd2 += TCBmTDB
            else:
                raise (ValueError("Invalid timescale, must be TDB or TCB."))

        result = self.calc(jd1, jd2, self.librat_ptr, rate)
        return result

    def TTmTDB(self, tt_jd, rate=False):
        """
        Time difference between TT and TDB.

        Interpolated using Chebyshev polynomials for an ephemeris file in
        TDB time that contains the time scale transformation 
        (self.timescale == "TDB" and self.has_time).

        Parameters
        ----------
        tt_jd :   float
                  Julian time in the TT (terrestrial time) timescale.
        rate :    bool
                  whether to return rate as well. Default is no.

        Returns
        -------
        float
                  The difference TT-TDB for the TT time, given in seconds.
                  If rate = True: also the clock rate in seconds per second
        """
        tt_jd1, tt_jd2 = Inpop.jd_unpack(tt_jd)

        if self.timescale == "TDB":
            if self.has_time:
                result = self.calc(tt_jd1, tt_jd2, self.TTmTDB_ptr, rate)[... , [0]]
                return result[... , 0]
        raise (KeyError("Ephemeris lacks TTmTDB transform"))

    def TCGmTCB(self, tcg_jd, rate=False):
        """
        Time difference between TCG and TCB.

        Only available for an ephemeris file in TCB time that contains time
        scale transformation data (self.timescale == "TCB" and self.has_time). 

        Parameters
        ----------
        tcg_jd :  float
                  Julian time in the TCG (geocentric coordinate time) timescale.
        rate :    bool
                  whether to return rate as well. Default is no.


        Returns
        -------
        float
                The difference TCG-TDB for the TCG time, given in seconds.
                If rate = True: also the clock rate in seconds per second

        """
        tcg_jd1, tcg_jd2 = Inpop.jd_unpack(tcg_jd)

        if self.timescale == "TCB":
            if self.has_time:
                result = self.calc(tcg_jd1, tcg_jd2, self.TTmTDB_ptr, rate)[... , [0]]
                return result[... , 0]
        raise (KeyError("Ephemeris lacks TCGmTCB transform"))

    def close(self):
        """
        Close the INPOP file.

        Returns
        -------
        None.

        """
        if self.file:
            self.file.close()
        self.file = None

    def __del__(self):
        """
        Destructor, closes the INPOP file (if open).

        Returns
        -------
        None.

        """
        self.close()
