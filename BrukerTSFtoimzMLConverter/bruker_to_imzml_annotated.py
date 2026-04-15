"""
bruker_to_imzml.py
==================
Convert a Bruker timsTOF MALDI imaging dataset (.d directory, TSF format)
to imzML + .ibd (processed mode, one m/z + intensity array per pixel).

Designed for non-PASEF QTOF imaging data (ScanMode=20, MsMsType=0).
Works on Windows and Linux; requires the Bruker timsdata shared library
(timsdata.dll / libtimsdata.so) from the TDF-SDK.

--- FORMAT BACKGROUND ---

A Bruker .d directory for MALDI imaging contains two key files:

  analysis.tsf   — SQLite database holding all metadata:
                     • GlobalMetadata  : instrument/acquisition settings as key-value pairs
                     • Frames          : one row per laser shot (spectrum), with TIC,
                                         BPC, peak count, temperature, calibration ID, etc.
                     • MaldiFrameInfo  : maps each Frame to a stage XY position or spot name
                     • CalibrationInfo : polynomial coefficients for m/z axis

  analysis.tsf_bin — raw binary blob of spectral data, accessed only via the
                     Bruker SDK (not directly readable). The SDK handles decompression
                     and returns arrays of (bin-index, intensity) pairs.

The SDK converts bin indices → m/z using per-frame calibration polynomials stored
in CalibrationInfo. You must call tsf_index_to_mz(); do NOT try to build the
calibration yourself from the SQLite tables.

--- imzML FORMAT BACKGROUND ---

imzML is a two-file format:
  .imzML  — XML metadata describing every spectrum (pixel coordinates, array
             sizes, byte offsets into the binary file, CV-term annotations)
  .ibd    — raw binary file containing all m/z and intensity arrays end-to-end

"Processed mode" means each pixel has its own independent m/z axis (vs.
"continuous mode" where all pixels share one common m/z axis). Processed mode
is correct for centroided MALDI data because each pixel has different peaks.

CV terms (controlled vocabulary) are standardised identifiers from the PSI-MS
ontology (MS:xxxxxxx) and the Imaging MS ontology (IMS:xxxxxxx). Every piece
of metadata in the imzML XML must reference the appropriate CV term so that
downstream tools (Cardinal, MSiReader, SCiLS, etc.) can parse it correctly.

--- DEPENDENCIES ---
    pip install numpy tqdm

The Bruker SDK library must be either:
  - placed next to this script (or on PATH / LD_LIBRARY_PATH), OR
  - its full path supplied via the `sdk_path` argument.

--- USAGE (Jupyter) ---
    from bruker_to_imzml import BrukerToImzML

    converter = BrukerToImzML(
        bruker_d_path   = "/path/to/my_dataset.d",
        output_dir      = "/path/to/output",          # imzML + ibd written here
        output_name     = "my_dataset",               # base filename (no extension)
        sdk_path        = None,                       # or "/path/to/libtimsdata.so"
        spectrum_type   = "centroid",                 # "centroid" or "profile"
        use_recalibrated= True,                       # use post-acquisition recalibration if available
    )
    converter.convert()

--- OUTPUT ---
    output_dir/
        my_dataset.imzML    — XML metadata file
        my_dataset.ibd      — binary spectral data (processed mode)
"""

import os
import sys
import uuid        # for generating a unique run ID in the imzML XML
import struct      # available for manual binary packing if needed (not used directly here)
import hashlib     # SHA-1 checksum of the .ibd file (embedded in imzML for validation)
import sqlite3     # direct access to the analysis.tsf SQLite database
import datetime    # timestamp for the imzML file header
import numpy as np
from pathlib import Path
from ctypes import (
    # ctypes lets Python call functions in compiled C shared libraries (.dll / .so).
    # We use it here to call the Bruker timsdata SDK, which has no native Python bindings —
    # it exposes a C API that we bind manually.
    cdll,              # loads the shared library
    c_char_p,          # C type: char*  (used for string arguments, e.g. file paths)
    c_uint32,          # C type: unsigned 32-bit int
    c_uint64,          # C type: unsigned 64-bit int (used for the SDK "handle")
    c_int32,           # C type: signed 32-bit int
    c_int64,           # C type: signed 64-bit int (used for frame IDs)
    c_float,           # C type: 32-bit float (intensity arrays)
    c_double,          # C type: 64-bit double (m/z and index arrays)
    POINTER,           # constructs a pointer type, e.g. POINTER(c_double) = double*
    create_string_buffer,  # allocates a mutable bytes buffer for C to write into
)

try:
    from tqdm.auto import tqdm   # progress bar; auto selects notebook vs terminal display
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[bruker_to_imzml] tqdm not found — install it for a progress bar: pip install tqdm")


# =============================================================================
# SECTION 1: SDK LOADER
# =============================================================================
# The Bruker timsdata SDK is a native C shared library. Python cannot call it
# directly — we use ctypes to:
#   1. Load the library into memory (cdll.LoadLibrary)
#   2. Declare the argument types (argtypes) and return type (restype) for each
#      function we want to call. This is mandatory; without it, ctypes cannot
#      correctly marshal Python values into C types and back.
# =============================================================================

def _load_sdk(sdk_path=None):
    """
    Load the Bruker timsdata shared library and bind all functions needed for
    TSF (non-PASEF MALDI imaging) data access.

    The SDK exposes a C API. Every function we call must have its argtypes and
    restype declared here so ctypes knows how to convert Python objects to C
    types when making the call, and how to interpret the return value.

    Parameters
    ----------
    sdk_path : str or Path or None
        Explicit path to the .dll / .so file. If None, the OS searches its
        default library paths (PATH on Windows, LD_LIBRARY_PATH on Linux).

    Returns
    -------
    lib : ctypes CDLL object
        The loaded library with all function signatures bound.
    """
    if sdk_path is not None:
        # Load from an explicit path supplied by the user.
        lib = cdll.LoadLibrary(str(sdk_path))
    else:
        # No explicit path — let the OS find the library by name.
        # On Windows: timsdata.dll must be on PATH or in the working directory.
        # On Linux:   libtimsdata.so must be on LD_LIBRARY_PATH or in /usr/lib etc.
        if sys.platform.startswith("win"):
            libname = "timsdata.dll"
        elif sys.platform.startswith("linux"):
            libname = "libtimsdata.so"
        else:
            raise OSError(f"Unsupported platform: {sys.platform}")
        lib = cdll.LoadLibrary(libname)

    # --- Lifecycle functions ---
    # tsf_open: opens a .d directory and returns an opaque integer "handle".
    # All subsequent SDK calls take this handle as their first argument.
    # Returns 0 on failure.
    lib.tsf_open.argtypes = [c_char_p, c_uint32]  # (directory path, use_recalibrated flag)
    lib.tsf_open.restype  = c_uint64               # handle (0 = failure)

    # tsf_close: releases all SDK resources for this handle. Must always be called.
    lib.tsf_close.argtypes = [c_uint64]  # (handle,)
    lib.tsf_close.restype  = None

    # tsf_get_last_error_string: retrieves the last error message from the SDK.
    # Called in two passes: first with (None, 0) to learn the required buffer size,
    # then with a real buffer of that size to get the actual string.
    lib.tsf_get_last_error_string.argtypes = [c_char_p, c_uint32]  # (buf, buf_len)
    lib.tsf_get_last_error_string.restype  = c_uint32               # required length

    # tsf_has_recalibrated_state: returns 1 if post-acquisition recalibration
    # data exists in this dataset (e.g., from Bruker DataAnalysis software).
    lib.tsf_has_recalibrated_state.argtypes = [c_uint64]  # (handle,)
    lib.tsf_has_recalibrated_state.restype  = c_uint32    # 0 or 1

    # --- Spectrum reader: centroid (line spectrum) ---
    # tsf_read_line_spectrum_v2: reads a peak-picked (centroided) spectrum for one frame.
    # The SDK returns bin indices (not m/z values) in index_array — these must be
    # converted to m/z separately using tsf_index_to_mz.
    # Returns -1 on error, or the number of peaks (may exceed `length` if buffer too small).
    lib.tsf_read_line_spectrum_v2.argtypes = [
        c_uint64,          # handle
        c_int64,           # spectrum_id (= Frames.Id in the SQLite database)
        POINTER(c_double), # index_array: output buffer for bin indices (caller-allocated)
        POINTER(c_float),  # intensity_array: output buffer for peak intensities
        c_int32            # length: capacity of the output buffers (in elements, not bytes)
    ]
    lib.tsf_read_line_spectrum_v2.restype = c_int32  # actual peak count, or -1

    # --- Spectrum reader: profile ---
    # tsf_read_profile_spectrum_v2: reads a raw profile spectrum for one frame.
    # Returns a dense intensity array where the array index IS the bin index.
    # (No separate index array — every bin from 0..N-1 is present.)
    # Returns -1 on error, or the number of profile points.
    lib.tsf_read_profile_spectrum_v2.argtypes = [
        c_uint64,          # handle
        c_int64,           # spectrum_id
        POINTER(c_uint32), # profile_array: output buffer for intensities (uint32 for profile)
        c_int32            # length: capacity of the buffer
    ]
    lib.tsf_read_profile_spectrum_v2.restype = c_int32  # actual profile length, or -1

    # --- m/z conversion ---
    # tsf_index_to_mz: converts an array of (possibly non-integer) bin indices to
    # calibrated m/z values. The calibration is per-frame (each frame may have
    # slightly different calibration coefficients stored in CalibrationInfo).
    # This is a vectorised call — pass arrays of N indices, get N m/z values back.
    _conv_args = [
        c_uint64,          # handle
        c_int64,           # frame_id (calibration is per-frame)
        POINTER(c_double), # in:  input array of bin indices
        POINTER(c_double), # out: output array of m/z values (caller-allocated)
        c_uint32           # cnt: number of values to convert
    ]
    lib.tsf_index_to_mz.argtypes = _conv_args
    lib.tsf_index_to_mz.restype  = c_uint32  # 1 on success, 0 on failure

    return lib


def _tsf_last_error(lib):
    """
    Retrieve the last error string from the SDK (thread-local).

    The SDK uses a two-pass pattern to return variable-length strings:
      Pass 1: call with (None, 0) — SDK returns the required buffer size.
      Pass 2: call with a real buffer of that size — SDK fills it with the message.

    Returns the error as a Python str.
    """
    n   = lib.tsf_get_last_error_string(None, 0)  # query required buffer length
    buf = create_string_buffer(n)                  # allocate mutable byte buffer
    lib.tsf_get_last_error_string(buf, n)          # fill buffer with error message
    return buf.value.decode("utf-8", errors="replace")


# =============================================================================
# SECTION 2: LOW-LEVEL SPECTRUM READERS
# =============================================================================
# These functions handle the "buffer-growing loop" pattern required by the SDK.
# The SDK cannot tell us in advance how many peaks a spectrum contains, so we:
#   1. Allocate a buffer of a guessed size.
#   2. Call the SDK reader.
#   3. If the SDK reports it needed more space than we gave it, grow the buffer
#      and retry. This loop converges in at most 2 iterations for most spectra.
# =============================================================================

def _read_centroid(lib, handle, frame_id, buf_size=4096):
    """
    Read a peak-picked (centroided) spectrum for one frame.

    The SDK returns bin indices and intensities separately. Bin indices are
    floating-point values (not integers) because the SDK interpolates peak
    positions within a bin for sub-bin accuracy. We must convert them to
    physical m/z values using tsf_index_to_mz before storing.

    Parameters
    ----------
    lib      : ctypes CDLL  — loaded Bruker SDK
    handle   : int          — open dataset handle from tsf_open
    frame_id : int          — Frames.Id from the SQLite database
    buf_size : int          — initial buffer capacity (grows automatically if needed)

    Returns
    -------
    mzs  : np.ndarray[float64]  — calibrated m/z values, one per peak
    ints : np.ndarray[float32]  — peak intensities, matching length to mzs
    """
    while True:
        # Pre-allocate numpy arrays that the SDK will write into via raw pointers.
        # np.empty is used (not np.zeros) for speed — we overwrite every element.
        idx_buf = np.empty(buf_size, dtype=np.float64)  # bin indices (SDK output)
        int_buf = np.empty(buf_size, dtype=np.float32)  # intensities (SDK output)

        # Call the SDK. We pass C pointers to the numpy array data buffers.
        # .ctypes.data_as(POINTER(c_double)) gives the SDK a raw double* pointer
        # into the numpy array's memory — no copy is made.
        n = lib.tsf_read_line_spectrum_v2(
            handle, frame_id,
            idx_buf.ctypes.data_as(POINTER(c_double)),
            int_buf.ctypes.data_as(POINTER(c_float)),
            buf_size
        )

        if n < 0:
            # SDK signals error with a negative return value.
            raise RuntimeError(f"tsf_read_line_spectrum_v2 failed: {_tsf_last_error(lib)}")

        if n > buf_size:
            # Buffer was too small. The SDK tells us exactly how many elements it
            # needed (n). Grow the buffer and retry. This path is uncommon —
            # the default buf_size=4096 covers most spectra.
            buf_size = n
            continue

        # Success: n peaks were written into the first n elements of each buffer.
        if n == 0:
            # Empty spectrum (no peaks above threshold). Return empty arrays.
            return np.empty(0, np.float64), np.empty(0, np.float32)

        # Slice only the valid n elements and make a copy so the original buffer
        # can be reused in the next loop iteration without aliasing issues.
        indices = idx_buf[:n].copy()

        # Convert bin indices → physical m/z using the per-frame calibration.
        # This is a vectorised C call — much faster than a Python loop.
        mzs = np.empty(n, dtype=np.float64)
        ok = lib.tsf_index_to_mz(
            handle, frame_id,
            indices.ctypes.data_as(POINTER(c_double)),
            mzs.ctypes.data_as(POINTER(c_double)),
            n
        )
        if not ok:
            raise RuntimeError(f"tsf_index_to_mz failed: {_tsf_last_error(lib)}")

        return mzs, int_buf[:n].copy()


def _read_profile(lib, handle, frame_id, buf_size=65536):
    """
    Read a raw profile (non-centroided) spectrum for one frame.

    Unlike the centroid reader, a profile spectrum is a dense array where
    position i holds the intensity at bin i. There is no separate index array —
    we create bin indices as a simple 0..N-1 integer sequence, then convert
    the whole axis to m/z in one vectorised call.

    Profile spectra are much larger than centroided ones (typically 50k–200k
    bins vs. a few hundred peaks), hence the larger default buf_size.

    Parameters
    ----------
    lib      : ctypes CDLL  — loaded Bruker SDK
    handle   : int          — open dataset handle
    frame_id : int          — Frames.Id
    buf_size : int          — initial buffer capacity (grows automatically)

    Returns
    -------
    mzs  : np.ndarray[float64]  — m/z value for every profile bin
    ints : np.ndarray[float32]  — intensity at each bin (cast from uint32)
    """
    while True:
        # Profile intensities are stored as uint32 in the SDK.
        int_buf = np.empty(buf_size, dtype=np.uint32)

        n = lib.tsf_read_profile_spectrum_v2(
            handle, frame_id,
            int_buf.ctypes.data_as(POINTER(c_uint32)),
            buf_size
        )

        if n < 0:
            raise RuntimeError(f"tsf_read_profile_spectrum_v2 failed: {_tsf_last_error(lib)}")

        if n > buf_size:
            buf_size = n
            continue

        if n == 0:
            return np.empty(0, np.float64), np.empty(0, np.float32)

        # For a profile spectrum, bin indices are simply 0, 1, 2, ..., n-1.
        # We create this array as float64 because tsf_index_to_mz expects doubles.
        indices = np.arange(n, dtype=np.float64)
        mzs     = np.empty(n, dtype=np.float64)

        ok = lib.tsf_index_to_mz(
            handle, frame_id,
            indices.ctypes.data_as(POINTER(c_double)),
            mzs.ctypes.data_as(POINTER(c_double)),
            n
        )
        if not ok:
            raise RuntimeError(f"tsf_index_to_mz failed: {_tsf_last_error(lib)}")

        # Cast to float32 for storage consistency with the centroid path.
        # Profile intensities from the SDK are uint32 but imzML typically
        # uses float32 for intensities.
        return mzs, int_buf[:n].astype(np.float32)


# =============================================================================
# SECTION 3: imzML CV TERM CONSTANTS
# =============================================================================
# imzML uses "controlled vocabulary" (CV) terms to annotate every piece of
# metadata in the XML. Each term has a unique accession number from either:
#   MS:xxxxxxx  — PSI-MS ontology (general mass spectrometry terms)
#   IMS:xxxxxxx — Imaging MS ontology (spatial imaging-specific terms)
#   UO:xxxxxxx  — Unit Ontology (physical units like micrometers)
#
# These constants make the XML-building code below readable and protect against
# typos in accession strings.
# =============================================================================

# Array type identifiers — tell readers what kind of data is in each binary block
CV_MZ_ARRAY          = "MS:1000514"   # this binary array contains m/z values
CV_INTENSITY_ARRAY   = "MS:1000515"   # this binary array contains intensity values

# Data encoding — tell readers the numeric type of each array element
CV_64BIT_FLOAT       = "MS:1000523"   # elements are 64-bit IEEE 754 doubles (8 bytes each)
CV_32BIT_FLOAT       = "MS:1000521"   # elements are 32-bit IEEE 754 floats  (4 bytes each)
CV_32BIT_INT         = "MS:1000519"   # elements are 32-bit signed integers   (4 bytes each)

# Compression
CV_NO_COMPRESSION    = "MS:1000576"   # binary data is stored uncompressed
CV_ZLIB              = "MS:1000574"   # binary data is zlib-compressed (not used here)

# Spectrum type
CV_CENTROID          = "MS:1000127"   # spectrum contains centroided (peak-picked) data
CV_PROFILE           = "MS:1000128"   # spectrum contains raw profile data

# Ion polarity
CV_POSITIVE          = "MS:1000130"   # positive ion mode
CV_NEGATIVE          = "MS:1000129"   # negative ion mode

# Ionisation source
CV_MALDI             = "MS:1000075"   # matrix-assisted laser desorption/ionisation

# Imaging geometry — pixel grid dimensions
CV_PIXEL_SIZE_X      = "IMS:1000046" # physical size of one pixel in x direction (µm)
CV_PIXEL_SIZE_Y      = "IMS:1000047" # physical size of one pixel in y direction (µm)
CV_MAX_COUNT_X       = "IMS:1000042" # total number of pixels in x direction
CV_MAX_COUNT_Y       = "IMS:1000043" # total number of pixels in y direction

# Pixel coordinates — per-spectrum spatial position
CV_PIXEL_COORD_X     = "IMS:1000050" # x index of this spectrum's pixel (1-based)
CV_PIXEL_COORD_Y     = "IMS:1000051" # y index of this spectrum's pixel (1-based)

# External binary data (processed mode imzML)
# In processed mode, binary data lives in the .ibd file, not inline in the XML.
# These three terms tell readers where in the .ibd file each array lives.
CV_EXTERNAL_DATA     = "IMS:1000101" # flag: data is in an external binary file
CV_EXTERNAL_OFFSET   = "IMS:1000102" # byte offset into the .ibd where this array starts
CV_EXTERNAL_LENGTH   = "IMS:1000103" # number of elements in this array

# Storage mode
CV_PROCESSED_MODE    = "IMS:1000031" # each pixel has its own m/z axis (vs continuous)

# m/z range — per-spectrum observed limits (used for efficient ion image extraction)
CV_MZ_MIN            = "MS:1000528"  # lowest observed m/z in this spectrum
CV_MZ_MAX            = "MS:1000527"  # highest observed m/z in this spectrum

CV_SCAN_POLARITY     = "MS:1000465"  # scan polarity (general)


def _cv(accession, name, value=None, unit_accession=None, unit_name=None):
    """
    Build a cvParam XML element string.

    cvParam elements are the standard way to attach controlled-vocabulary
    metadata to elements in mzML / imzML. Example output:
        <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>

    Parameters
    ----------
    accession      : str  — CV term accession, e.g. "MS:1000514"
    name           : str  — human-readable term name
    value          : any  — optional value for the parameter
    unit_accession : str  — optional Unit Ontology accession for the value's unit
    unit_name      : str  — optional unit name (e.g. "micrometer")
    """
    s = f'        <cvParam cvRef="MS" accession="{accession}" name="{name}"'
    if value is not None:
        s += f' value="{value}"'
    if unit_accession:
        s += f' unitCvRef="UO" unitAccession="{unit_accession}" unitName="{unit_name}"'
    s += '/>'
    return s


# =============================================================================
# SECTION 4: MAIN CONVERTER CLASS
# =============================================================================

class BrukerToImzML:
    """
    Convert a Bruker timsTOF MALDI imaging .d directory (TSF format) to imzML.

    The conversion pipeline has three stages:
      1. _load_metadata()  — read instrument/acquisition metadata from SQLite
      2. _load_frames()    — build a list of pixel descriptors (id, x, y, polarity)
      3. _write_ibd()      — read every spectrum via SDK, write binary .ibd file
      4. _write_imzml()    — write the XML .imzML file pointing into the .ibd

    Parameters
    ----------
    bruker_d_path : str or Path
        Path to the .d directory (must contain analysis.tsf).
    output_dir : str or Path
        Directory where the output files will be written.
    output_name : str, optional
        Base name for output files (default: stem of bruker_d_path).
    sdk_path : str or Path or None
        Full path to timsdata.dll / libtimsdata.so.
        If None, the library is searched on the system path.
    spectrum_type : str
        "centroid"  → use tsf_read_line_spectrum_v2  (peak-picked, smaller files)
        "profile"   → use tsf_read_profile_spectrum_v2 (raw profile, larger files)
    use_recalibrated : bool
        If True, use post-acquisition recalibration if available. Recommended.
    ms_level_filter : int or None
        Only export frames with this MsMsType (0=MS1, 2=MS2).
        None = export all frames regardless of type.
    """

    def __init__(
        self,
        bruker_d_path,
        output_dir      = None,
        output_name     = None,
        sdk_path        = None,
        spectrum_type   = "centroid",
        use_recalibrated= True,
        ms_level_filter = 0,
    ):
        self.d_path          = Path(bruker_d_path)
        # Default output location is the parent of the .d directory
        self.output_dir      = Path(output_dir) if output_dir else self.d_path.parent
        # Default output name is the .d directory stem (e.g. "my_dataset" from "my_dataset.d")
        self.output_name     = output_name or self.d_path.stem
        self.sdk_path        = sdk_path
        self.spectrum_type   = spectrum_type.lower()
        self.use_recalibrated= use_recalibrated
        self.ms_level_filter = ms_level_filter

        if self.spectrum_type not in ("centroid", "profile"):
            raise ValueError("spectrum_type must be 'centroid' or 'profile'")

        # Validate that this is a TSF dataset (non-PASEF QTOF imaging).
        # timsTOF with ion mobility uses analysis.tdf instead — a completely
        # different format that this converter does not handle.
        tsf_path = self.d_path / "analysis.tsf"
        if not tsf_path.exists():
            raise FileNotFoundError(
                f"No analysis.tsf found in {self.d_path}.\n"
                "Make sure this is a timsTOF TSF dataset (non-PASEF QTOF imaging). "
                "If you have a timsTOF with ion mobility, use analysis.tdf instead."
            )

        # Create output directory tree if needed (parents=True handles nested paths)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.imzml_path = self.output_dir / f"{self.output_name}.imzML"
        self.ibd_path   = self.output_dir / f"{self.output_name}.ibd"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def convert(self):
        """
        Run the full conversion pipeline and return the output file paths.

        Pipeline:
          1. Load the SDK and open the dataset.
          2. Read metadata and frame list from the SQLite database.
          3. Write the .ibd binary file (all spectra, one after another).
          4. Write the .imzML XML file (pointing into the .ibd via byte offsets).

        The SDK handle is always closed in a finally block to avoid resource leaks,
        even if an exception occurs during conversion.

        Returns
        -------
        (imzml_path, ibd_path) : tuple of str
        """
        print(f"[bruker_to_imzml] Opening: {self.d_path}")
        lib = _load_sdk(self.sdk_path)

        # tsf_open returns an opaque integer handle (like a file descriptor).
        # We pass 1 to request recalibrated data if available, 0 for raw calibration.
        handle = lib.tsf_open(
            str(self.d_path).encode("utf-8"),  # SDK expects UTF-8 bytes, not a Python str
            1 if self.use_recalibrated else 0
        )
        if handle == 0:
            # SDK returns 0 on failure; retrieve the error message for diagnosis.
            raise RuntimeError(f"Failed to open dataset: {_tsf_last_error(lib)}")

        # Report whether recalibration data is actually present (it's optional —
        # not all datasets go through DataAnalysis post-processing).
        recal = lib.tsf_has_recalibrated_state(handle)
        print(f"[bruker_to_imzml] Recalibrated state available: {bool(recal)}, "
              f"using: {self.use_recalibrated and bool(recal)}")

        try:
            # Open a separate SQLite connection for metadata queries.
            # The SDK and SQLite work independently: SDK reads the binary .tsf_bin;
            # SQLite reads the .tsf database. Both can be open simultaneously.
            conn   = sqlite3.connect(str(self.d_path / "analysis.tsf"))
            meta   = self._load_metadata(conn)
            frames = self._load_frames(conn)
            conn.close()  # done with metadata; close connection before writing output

            if len(frames) == 0:
                raise RuntimeError("No matching frames found. Check ms_level_filter setting.")

            print(f"[bruker_to_imzml] Found {len(frames)} spectra to convert")
            print(f"[bruker_to_imzml] Spectrum type: {self.spectrum_type}")
            print(f"[bruker_to_imzml] Writing ibd: {self.ibd_path}")

            # Stage 1: write the binary .ibd, collecting per-spectrum byte offsets
            spectrum_metadata = self._write_ibd(lib, handle, frames)

            # Stage 2: write the XML .imzML referencing those offsets
            print(f"[bruker_to_imzml] Writing imzML: {self.imzml_path}")
            self._write_imzml(meta, frames, spectrum_metadata)

        finally:
            # Always close the SDK handle to release file locks and memory.
            # This runs even if an exception propagates upward.
            lib.tsf_close(handle)

        print(f"\n[bruker_to_imzml] Done!")
        print(f"  imzML : {self.imzml_path}")
        print(f"  ibd   : {self.ibd_path}")
        return str(self.imzml_path), str(self.ibd_path)

    # ------------------------------------------------------------------
    # Stage 0: Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self, conn):
        """
        Read acquisition metadata from the analysis.tsf SQLite database.

        Metadata is spread across several tables. We collect everything into a
        flat dict and use it later to populate imzML header fields (pixel size,
        instrument name, etc.). We use broad try/except blocks because optional
        tables (MaldiApplicationInfo, ScanMode) may not exist in all datasets.

        Parameters
        ----------
        conn : sqlite3.Connection

        Returns
        -------
        meta : dict[str, Any]
            Flat key-value store of all metadata found.
        """
        meta = {}

        # GlobalMetadata holds the most important settings as simple key-value rows:
        # schema version, instrument serial number, acquisition software version,
        # polarity, mass range, laser settings, etc.
        for key, val in conn.execute("SELECT Key, Value FROM GlobalMetadata"):
            meta[key] = val

        # MaldiFrameInfo links each spectrum to a physical stage position.
        # It exists for imaging datasets but may not be present in LC-MS data.
        # We read just the first row to harvest any geometry-related column values
        # (laser spot width/height) that help us set pixel size in the imzML.
        try:
            # PRAGMA table_info returns column metadata: (cid, name, type, ...)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(MaldiFrameInfo)")]
            if cols:
                first = conn.execute("SELECT * FROM MaldiFrameInfo LIMIT 1").fetchone()
                if first:
                    for col, val in zip(cols, first):
                        meta[f"MaldiFrameInfo.{col}"] = val
        except Exception:
            pass  # table may not exist; silently skip

        # Some datasets store additional scan parameters in auxiliary tables.
        for table in ("ScanMode", "MaldiApplicationInfo"):
            try:
                for key, val in conn.execute(f"SELECT Key, Value FROM {table}"):
                    meta[f"{table}.{key}"] = val
            except Exception:
                pass

        return meta

    def _load_frames(self, conn):
        """
        Build the list of spectra to convert, including their pixel coordinates.

        Each Bruker frame corresponds to one laser shot = one pixel in the image.
        We need three things per frame:
          1. Frame ID — to call the SDK spectrum reader
          2. (x, y)   — pixel grid coordinates for the imzML spatial index
          3. Polarity, scan mode, peak count — for imzML metadata

        Pixel coordinates come from the MaldiFrameInfo table, which may store
        them as numeric grid indices (XIndexPos / YIndexPos — most common) or
        as alphanumeric spot names (SpotName, e.g. "R01C02" — older formats).

        Parameters
        ----------
        conn : sqlite3.Connection

        Returns
        -------
        frames : list of dict
            One dict per spectrum, with keys: id, x, y, polarity, scan_mode,
            msms_type, num_peaks, tic, bpc.
        """
        # Discover which coordinate columns are present in MaldiFrameInfo
        # so we can pick the right coordinate-loading strategy below.
        maldi_cols = []
        try:
            maldi_cols = [r[1] for r in conn.execute("PRAGMA table_info(MaldiFrameInfo)")]
        except Exception:
            pass

        # Query the Frames table for all spectra that:
        #   - Have at least one ion detected (SummedIntensities > 0)
        #   - Match the requested MS level (MsMsType filter)
        # Order by Id ensures the .ibd file is written in acquisition order,
        # which is the natural order for most downstream tools.
        frames_query = """
            SELECT f.Id, f.Polarity, f.ScanMode, f.MsMsType, f.NumPeaks,
                   f.SummedIntensities, f.MaxIntensity
            FROM Frames f
            WHERE f.SummedIntensities > 0
        """
        if self.ms_level_filter is not None:
            # MsMsType=0 means MS1; MsMsType=2 means MS2 (fragmentation).
            # For MALDI imaging we almost always want MsMsType=0.
            frames_query += f" AND f.MsMsType = {self.ms_level_filter}"
        frames_query += " ORDER BY f.Id"

        rows = conn.execute(frames_query).fetchall()

        # --- Coordinate resolution: try three strategies in order ---

        coord_map = {}  # frame_id → (x, y) in pixel grid coordinates

        if "XIndexPos" in maldi_cols and "YIndexPos" in maldi_cols:
            # Best case: numeric grid indices are stored directly.
            # XIndexPos and YIndexPos are 1-based pixel grid coordinates.
            for fid, x, y in conn.execute(
                "SELECT Frame, XIndexPos, YIndexPos FROM MaldiFrameInfo"
            ):
                coord_map[fid] = (int(x), int(y))

        elif "SpotName" in maldi_cols:
            # Older datasets use alphanumeric spot names like "R01C01".
            # Parse these into (x=col, y=row) integer pairs.
            for fid, spot in conn.execute(
                "SELECT Frame, SpotName FROM MaldiFrameInfo"
            ):
                coord_map[fid] = _parse_spot_name(spot)

        else:
            # No coordinate information available (unusual — should not happen
            # for well-formed imaging datasets).
            # Fall back to a 1D layout: all pixels in row y=1, sequential x.
            print("[bruker_to_imzml] WARNING: No pixel coordinate info found in MaldiFrameInfo. "
                  "Assigning sequential x coordinates (1-based).")
            for i, (fid, *_) in enumerate(rows):
                coord_map[fid] = (i + 1, 1)

        # Assemble the final frame list, joining coordinate data with frame metadata.
        frames = []
        for (fid, polarity, scan_mode, msms_type, num_peaks, summed, maxint) in rows:
            x, y = coord_map.get(fid, (fid, 1))  # fallback if frame missing from MaldiFrameInfo
            frames.append({
                "id":        fid,
                "x":         x,           # pixel column (1-based, left to right)
                "y":         y,           # pixel row    (1-based, top to bottom)
                "polarity":  polarity,    # "+" or "-"
                "scan_mode": scan_mode,   # 20 = MALDI for imaging data
                "msms_type": msms_type,   # 0 = MS1
                "num_peaks": num_peaks,   # from SQLite; used as a sanity check
                "tic":       summed,      # total ion current (sum of all intensities)
                "bpc":       maxint,      # base peak intensity (maximum intensity)
            })

        return frames

    # ------------------------------------------------------------------
    # Stage 1: Binary data (.ibd)
    # ------------------------------------------------------------------

    def _write_ibd(self, lib, handle, frames):
        """
        Write all spectra to the .ibd binary file and record per-spectrum offsets.

        imzML "processed mode" layout (what we use here):
          For each pixel, we write two consecutive blocks:
            [mz_array: N × float64][intensity_array: N × float32]
          The byte offset of each block is recorded and later written into the
          imzML XML so that readers can seek directly to any spectrum.

        We also compute a SHA-1 checksum of the entire .ibd file as we write it.
        This checksum is embedded in the imzML XML and can be used by tools to
        verify the file was not corrupted in transit.

        Parameters
        ----------
        lib    : ctypes CDLL  — loaded Bruker SDK
        handle : int          — open dataset handle
        frames : list of dict — frame descriptors from _load_frames()

        Returns
        -------
        spectrum_meta : list of dict
            One dict per spectrum with keys:
              mz_offset, mz_length, mz_count  — location of m/z array in .ibd
              int_offset, int_length, int_count — location of intensity array
              mz_min, mz_max                   — observed m/z range (for imzML header)
        """
        # Select the appropriate reader function based on spectrum_type setting.
        reader = _read_centroid if self.spectrum_type == "centroid" else _read_profile

        sha1          = hashlib.sha1()  # running SHA-1 hash; updated with each written block
        spectrum_meta = []

        # Optional progress bar (shows conversion speed and estimated time remaining)
        progress = (tqdm(total=len(frames), desc="Converting spectra", unit="px")
                    if HAS_TQDM else None)

        with open(self.ibd_path, "wb") as f:
            byte_offset = 0  # running byte position in the .ibd file

            for fr in frames:
                # Fetch this spectrum from the Bruker SDK.
                # mzs  : float64 array of calibrated m/z values
                # ints : float32 array of peak intensities
                mzs, ints = reader(lib, handle, fr["id"])

                # Serialise arrays to raw bytes in the correct dtypes for imzML.
                # m/z arrays: float64 (8 bytes/element) — high precision needed for mass accuracy
                # intensity arrays: float32 (4 bytes/element) — 32-bit sufficient for intensities
                mz_bytes  = mzs.astype(np.float64).tobytes()
                int_bytes = ints.astype(np.float32).tobytes()

                # Update running SHA-1 hash with the bytes we're about to write.
                sha1.update(mz_bytes)
                sha1.update(int_bytes)

                # Write m/z array and record its position in the file.
                # Note: mz_offset is captured BEFORE writing (it's the start position).
                mz_offset   = byte_offset
                f.write(mz_bytes)
                mz_length   = len(mz_bytes)   # total bytes (= mz_count * 8)
                byte_offset += mz_length

                # Write intensity array immediately after m/z array.
                int_offset   = byte_offset
                f.write(int_bytes)
                int_length   = len(int_bytes)  # total bytes (= int_count * 4)
                byte_offset += int_length

                # Store the offset/length metadata for this spectrum.
                # The imzML XML will reference these exact values.
                spectrum_meta.append({
                    "mz_offset":   mz_offset,         # byte start of m/z array in .ibd
                    "mz_length":   mz_length,          # byte length of m/z array
                    "mz_count":    len(mzs),           # number of m/z elements (= peaks)
                    "int_offset":  int_offset,         # byte start of intensity array in .ibd
                    "int_length":  int_length,         # byte length of intensity array
                    "int_count":   len(ints),          # number of intensity elements
                    "mz_min":      float(mzs.min()) if len(mzs) else 0.0,  # observed m/z range
                    "mz_max":      float(mzs.max()) if len(mzs) else 0.0,
                })

                if progress:
                    progress.update(1)

        if progress:
            progress.close()

        # Finalise the SHA-1 checksum and store it on self for use in _write_imzml.
        self._ibd_sha1 = sha1.hexdigest().upper()
        print(f"[bruker_to_imzml] ibd SHA-1: {self._ibd_sha1}")
        return spectrum_meta

    # ------------------------------------------------------------------
    # Stage 2: imzML XML
    # ------------------------------------------------------------------

    def _write_imzml(self, meta, frames, spectrum_meta):
        """
        Write the .imzML XML file.

        imzML is a specialised dialect of mzML (the standard proteomics MS format)
        with added imaging-specific elements. The XML structure is:

          <mzML>
            <cvList>           — declares which CV ontologies are used
            <fileDescription>  — source file provenance, storage mode
            <referenceableParamGroupList>  — reusable CV parameter groups
                                             (avoids repeating encoding info per spectrum)
            <softwareList>     — what software created this file
            <scanSettingsList> — imaging geometry (pixel grid size, pixel dimensions)
            <instrumentConfigurationList> — instrument description
            <dataProcessingList> — what processing was applied
            <run>
              <spectrumList>   — one <spectrum> element per pixel
                <spectrum>
                  <scanList>   — pixel (x,y) coordinates
                  <binaryDataArrayList>
                    <binaryDataArray>  — m/z array: offset + length into .ibd
                    <binaryDataArray>  — intensity array: offset + length into .ibd

        Parameters
        ----------
        meta          : dict        — metadata from _load_metadata()
        frames        : list of dict — frame descriptors from _load_frames()
        spectrum_meta : list of dict — per-spectrum offsets from _write_ibd()
        """

        # --- Imaging geometry ---
        xs    = [fr["x"] for fr in frames]
        ys    = [fr["y"] for fr in frames]
        max_x = max(xs)  # number of columns in the pixel grid
        max_y = max(ys)  # number of rows in the pixel grid

        # Pixel size in micrometres.
        # Try to read from metadata; fall back to 100 µm (a common MALDI step size).
        # LaserWidth / LaserHeight describe the laser spot footprint, which equals
        # the pixel size when raster step == spot size.
        pixel_size_x = float(meta.get("MaldiFrameInfo.LaserWidth",
                              meta.get("PixelSizeX", 100.0)))
        pixel_size_y = float(meta.get("MaldiFrameInfo.LaserHeight",
                              meta.get("PixelSizeY", pixel_size_x)))

        # --- Ion polarity ---
        # Assume polarity is consistent across the dataset (mixed-polarity datasets
        # are very unusual for MALDI imaging). Take polarity from the first frame.
        polarity      = frames[0]["polarity"]
        polarity_cv   = CV_POSITIVE if polarity == "+" else CV_NEGATIVE
        polarity_name = "positive scan" if polarity == "+" else "negative scan"

        # --- Spectrum encoding ---
        spectrum_cv   = CV_CENTROID if self.spectrum_type == "centroid" else CV_PROFILE
        spectrum_name = "centroid spectrum" if self.spectrum_type == "centroid" else "profile spectrum"

        # A UUID uniquely identifies this run — useful for downstream data provenance.
        run_uuid     = str(uuid.uuid4()).upper()
        now          = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        ibd_filename = self.ibd_path.name

        # Global m/z range across all spectra (used in some tools for display scaling).
        all_mz_min = min(s["mz_min"] for s in spectrum_meta if s["mz_count"] > 0)
        all_mz_max = max(s["mz_max"] for s in spectrum_meta if s["mz_count"] > 0)

        # Build the XML by accumulating lines in a list, then joining at the end.
        # This avoids repeated string concatenation (which is O(n²) in Python)
        # and keeps the structure easy to read.
        lines = []
        a = lines.append  # local alias — saves repeated attribute lookup in the loop

        # --- XML declaration and root element ---
        a('<?xml version="1.0" encoding="utf-8"?>')
        a('<mzML xmlns="http://psi.hupo.org/ms/mzml"')
        a('      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
        a('      xsi:schemaLocation="http://psi.hupo.org/ms/mzml '
          'http://psidev.info/files/ms/mzML/xsd/mzML1.1.0_idx.xsd"')
        a('      xmlns:cv="http://psi.hupo.org/ms/mzml"')
        a('      version="1.1">')

        # --- cvList: declare the ontologies we reference ---
        # Downstream parsers use these URIs to resolve CV term definitions.
        a('  <cvList count="3">')
        a('    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology"'
          ' version="4.1.30" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>')
        a('    <cv id="IMS" fullName="Mass Spectrometry Imaging Ontology"'
          ' version="1.1.0" URI="https://raw.githubusercontent.com/imzML/imzML/master/imagingMS.obo"/>')
        a('    <cv id="UO"  fullName="Unit Ontology"'
          ' version="09:04:2014" URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>')
        a('  </cvList>')

        # --- fileDescription: provenance and storage mode ---
        a('  <fileDescription>')
        a('    <fileContent>')
        # Declare whether this is centroid or profile data at the file level.
        a(f'      <cvParam cvRef="MS" accession="{spectrum_cv}" name="{spectrum_name}" value=""/>')
        # "processed" = each pixel has its own m/z axis in the .ibd.
        # This is required for centroided data where peak lists differ between pixels.
        a(f'      <cvParam cvRef="IMS" accession="{CV_PROCESSED_MODE}" name="processed" value=""/>')
        a('    </fileContent>')
        # Record where the raw data came from (the original .d directory).
        a('    <sourceFileList count="1">')
        a(f'      <sourceFile id="sf1" name="{self.d_path.name}" location="{self.d_path.parent.as_uri()}">')
        a('        <cvParam cvRef="MS" accession="MS:1000564" name="PSI mzData file" value=""/>')
        a('      </sourceFile>')
        a('    </sourceFileList>')
        a('  </fileDescription>')

        # --- referenceableParamGroupList ---
        # These named groups define the binary encoding for m/z and intensity arrays.
        # Each spectrum's <binaryDataArray> will reference these groups by ID,
        # avoiding the need to repeat the same four cvParam lines thousands of times.
        a('  <referenceableParamGroupList count="2">')

        # mzArray group: m/z values are stored as 64-bit floats (double precision),
        # uncompressed, in the external .ibd file.
        a('    <referenceableParamGroup id="mzArray">')
        a(f'      <cvParam cvRef="MS" accession="{CV_MZ_ARRAY}" name="m/z array" value=""/>')
        a(f'      <cvParam cvRef="MS" accession="{CV_64BIT_FLOAT}" name="64-bit float" value=""/>')
        a(f'      <cvParam cvRef="MS" accession="{CV_NO_COMPRESSION}" name="no compression" value=""/>')
        a(f'      <cvParam cvRef="IMS" accession="{CV_EXTERNAL_DATA}" name="external data" value="true"/>')
        a('    </referenceableParamGroup>')

        # intensityArray group: intensities are stored as 32-bit floats,
        # uncompressed, in the external .ibd file.
        a('    <referenceableParamGroup id="intensityArray">')
        a(f'      <cvParam cvRef="MS" accession="{CV_INTENSITY_ARRAY}" name="intensity array" value=""/>')
        a(f'      <cvParam cvRef="MS" accession="{CV_32BIT_FLOAT}" name="32-bit float" value=""/>')
        a(f'      <cvParam cvRef="MS" accession="{CV_NO_COMPRESSION}" name="no compression" value=""/>')
        a(f'      <cvParam cvRef="IMS" accession="{CV_EXTERNAL_DATA}" name="external data" value="true"/>')
        a('    </referenceableParamGroup>')

        a('  </referenceableParamGroupList>')

        # --- softwareList: records what created this file ---
        a('  <softwareList count="1">')
        a('    <software id="bruker_to_imzml" version="1.0">')
        a('      <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="bruker_to_imzml"/>')
        a('    </software>')
        a('  </softwareList>')

        # --- scanSettingsList: imaging geometry ---
        # This section defines the physical layout of the pixel grid.
        # max_x / max_y: the extent of the raster in pixels.
        # pixel_size: physical step size of the stage in micrometres.
        a('  <scanSettingsList count="1">')
        a('    <scanSettings id="scanSettings1">')
        a(f'      <cvParam cvRef="IMS" accession="{CV_MAX_COUNT_X}" name="max count of pixels x" value="{max_x}"/>')
        a(f'      <cvParam cvRef="IMS" accession="{CV_MAX_COUNT_Y}" name="max count of pixels y" value="{max_y}"/>')
        a(f'      <cvParam cvRef="IMS" accession="{CV_PIXEL_SIZE_X}" name="pixel size x" value="{pixel_size_x}"'
          ' unitCvRef="UO" unitAccession="UO:0000017" unitName="micrometer"/>')
        a(f'      <cvParam cvRef="IMS" accession="{CV_PIXEL_SIZE_Y}" name="pixel size y" value="{pixel_size_y}"'
          ' unitCvRef="UO" unitAccession="UO:0000017" unitName="micrometer"/>')
        a('    </scanSettings>')
        a('  </scanSettingsList>')

        # --- instrumentConfigurationList ---
        # Describes the ionisation source, mass analyser, and detector.
        # For timsTOF MALDI imaging: MALDI source → TOF analyser → electron multiplier detector.
        a('  <instrumentConfigurationList count="1">')
        a('    <instrumentConfiguration id="IC1">')
        a('      <cvParam cvRef="MS" accession="MS:1001535" name="Bruker Daltonics timsTOF series" value=""/>')
        a('      <componentList count="3">')
        a('        <source order="1">')
        a(f'          <cvParam cvRef="MS" accession="{CV_MALDI}" name="matrix-assisted laser desorption ionization" value=""/>')
        a('        </source>')
        a('        <analyzer order="2">')
        a('          <cvParam cvRef="MS" accession="MS:1000084" name="time-of-flight" value=""/>')
        a('        </analyzer>')
        a('        <detector order="3">')
        a('          <cvParam cvRef="MS" accession="MS:1000253" name="electron multiplier" value=""/>')
        a('        </detector>')
        a('      </componentList>')
        a('    </instrumentConfiguration>')
        a('  </instrumentConfigurationList>')

        # --- dataProcessingList ---
        # Records what transformations were applied to go from raw data to this file.
        # "peak picking" is added for centroid mode (the SDK applies a peak finder).
        # "Conversion to mzML" is always present (format conversion).
        a('  <dataProcessingList count="1">')
        a('    <dataProcessing id="dp1">')
        a('      <processingMethod order="1" softwareRef="bruker_to_imzml">')
        if self.spectrum_type == "centroid":
            a('        <cvParam cvRef="MS" accession="MS:1000035" name="peak picking" value=""/>')
        a('        <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>')
        a('      </processingMethod>')
        a('    </dataProcessing>')
        a('  </dataProcessingList>')

        # --- run / spectrumList ---
        # The main data section: one <spectrum> element per pixel.
        a(f'  <run defaultInstrumentConfigurationRef="IC1" id="{run_uuid}">')
        a(f'    <spectrumList count="{len(frames)}" defaultDataProcessingRef="dp1">')

        for i, (fr, sm) in enumerate(zip(frames, spectrum_meta)):
            # spec_id follows the mzML convention "spectrum=N" (1-based).
            spec_id  = f"spectrum={i+1}"
            scan_idx = i + 1  # index attribute is also 1-based

            a(f'      <spectrum id="{spec_id}" defaultArrayLength="{sm["mz_count"]}" index="{scan_idx}">')
            # Inherit array encoding from the referenceable param group (avoids repetition).
            a('        <referenceableParamGroupRef ref="mzArray"/>')
            a('        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>')
            a(f'        <cvParam cvRef="MS" accession="{spectrum_cv}" name="{spectrum_name}" value=""/>')
            a(f'        <cvParam cvRef="MS" accession="{polarity_cv}" name="{polarity_name}" value=""/>')
            # Per-spectrum m/z range — enables fast ion image extraction by range query.
            a(f'        <cvParam cvRef="MS" accession="{CV_MZ_MIN}" name="lowest observed m/z" value="{sm["mz_min"]:.6f}"/>')
            a(f'        <cvParam cvRef="MS" accession="{CV_MZ_MAX}" name="highest observed m/z" value="{sm["mz_max"]:.6f}"/>')

            # scanList: spatial metadata.
            # "no combination" means this spectrum represents a single scan (not an average).
            # The pixel (x, y) coordinates map this spectrum to its position in the image.
            a('        <scanList count="1">')
            a('          <cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>')
            a('          <scan>')
            a(f'            <cvParam cvRef="IMS" accession="{CV_PIXEL_COORD_X}" name="position x" value="{fr["x"]}"/>')
            a(f'            <cvParam cvRef="IMS" accession="{CV_PIXEL_COORD_Y}" name="position y" value="{fr["y"]}"/>')
            a('          </scan>')
            a('        </scanList>')

            # binaryDataArrayList: tells readers exactly where in the .ibd file to
            # find the m/z and intensity arrays for this pixel.
            # external offset = byte position in .ibd where this array starts
            # external length = number of elements (NOT bytes) in the array
            a('        <binaryDataArrayList count="2">')

            # m/z array pointer
            a('          <binaryDataArray>')
            a('            <referenceableParamGroupRef ref="mzArray"/>')
            a(f'            <cvParam cvRef="IMS" accession="{CV_EXTERNAL_OFFSET}" name="external offset" value="{sm["mz_offset"]}"/>')
            a(f'            <cvParam cvRef="IMS" accession="{CV_EXTERNAL_LENGTH}" name="external length" value="{sm["mz_count"]}"/>')
            a(f'            <cvParam cvRef="MS" accession="MS:1000786" name="non-standard data array" value=""/>')
            a('            <binary/>')  # empty: actual data is in the .ibd file
            a('          </binaryDataArray>')

            # intensity array pointer
            a('          <binaryDataArray>')
            a('            <referenceableParamGroupRef ref="intensityArray"/>')
            a(f'            <cvParam cvRef="IMS" accession="{CV_EXTERNAL_OFFSET}" name="external offset" value="{sm["int_offset"]}"/>')
            a(f'            <cvParam cvRef="IMS" accession="{CV_EXTERNAL_LENGTH}" name="external length" value="{sm["int_count"]}"/>')
            a(f'            <cvParam cvRef="MS" accession="MS:1000786" name="non-standard data array" value=""/>')
            a('            <binary/>')  # empty: actual data is in the .ibd file
            a('          </binaryDataArray>')

            a('        </binaryDataArrayList>')
            a('      </spectrum>')

        a('    </spectrumList>')
        a('  </run>')
        a('</mzML>')

        # Join all lines and write to disk in one operation (more efficient than
        # calling f.write() for each line in the loop above).
        with open(self.imzml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# =============================================================================
# SECTION 5: SPOT NAME PARSER
# =============================================================================
# Older Bruker MALDI datasets (and some microplate formats) identify each
# acquisition position by an alphanumeric spot name rather than numeric XY indices.
# This function parses the most common formats into (x, y) integer pixel coordinates.
# =============================================================================

def _parse_spot_name(spot):
    """
    Parse a Bruker MALDI spot name string into (x, y) pixel grid coordinates.

    Bruker uses several naming conventions depending on the instrument generation
    and acquisition method:

      "R01C01" / "r1c1"  — row/column format (most common for imaging rasters)
                           → x = column number, y = row number
      "0001", "0042"     — sequential integer (e.g. microplate spot index)
                           → x = integer value, y = 1 (all in one row)
      "A1", "B12"        — letter row + numeric column (microplate well format)
                           → x = column number, y = letter → integer (A=1, B=2, ...)

    In all cases, coordinates are returned as 1-based integers, consistent with
    the imzML convention (pixel coordinates start at 1, not 0).

    Parameters
    ----------
    spot : str or None  — the SpotName value from MaldiFrameInfo

    Returns
    -------
    (x, y) : tuple of int  — 1-based pixel grid coordinates
    """
    import re
    if spot is None:
        return (1, 1)
    spot = str(spot).strip()

    # Pattern: R##C## (case-insensitive)
    # Group 1 = row number, Group 2 = column number.
    # imzML convention: x = column (left-right), y = row (top-bottom).
    m = re.match(r"[Rr](\d+)[Cc](\d+)", spot)
    if m:
        return (int(m.group(2)), int(m.group(1)))  # (col=x, row=y)

    # Pattern: pure integer string (e.g. "0042")
    m = re.match(r"^(\d+)$", spot)
    if m:
        n = int(m.group(1))
        return (n, 1)  # treat as a 1D sequence along x-axis

    # Pattern: letter(s) + number (microplate well notation, e.g. "A1", "AB12")
    # Convert the letter prefix to a row index using base-26 arithmetic:
    #   A=1, B=2, ..., Z=26, AA=27, AB=28, ...
    m = re.match(r"([A-Za-z]+)(\d+)", spot)
    if m:
        row = sum((ord(c.upper()) - 64) * (26**i)
                  for i, c in enumerate(reversed(m.group(1))))
        col = int(m.group(2))
        return (col, row)

    # Unrecognised format: return origin so conversion can continue with a warning.
    return (1, 1)


# =============================================================================
# SECTION 6: CONVENIENCE FUNCTION FOR JUPYTER
# =============================================================================

def convert(
    bruker_d_path,
    output_dir      = None,
    output_name     = None,
    sdk_path        = None,
    spectrum_type   = "centroid",
    use_recalibrated= True,
    ms_level_filter = 0,
):
    """
    One-line helper for Jupyter notebooks. Creates a BrukerToImzML instance
    and immediately runs the conversion.

    Parameters
    ----------
    bruker_d_path    : str or Path  — path to the .d directory
    output_dir       : str or Path  — where to write output (default: .d parent)
    output_name      : str          — base filename without extension
    sdk_path         : str or Path  — explicit path to timsdata library
    spectrum_type    : str          — "centroid" or "profile"
    use_recalibrated : bool         — use post-acquisition recalibration if available
    ms_level_filter  : int or None  — 0=MS1 only, 2=MS2 only, None=all

    Returns
    -------
    (imzml_path, ibd_path) : tuple of str

    Example
    -------
        from bruker_to_imzml import convert
        imzml, ibd = convert("/data/my_dataset.d", output_dir="/data/output")
    """
    c = BrukerToImzML(
        bruker_d_path    = bruker_d_path,
        output_dir       = output_dir,
        output_name      = output_name,
        sdk_path         = sdk_path,
        spectrum_type    = spectrum_type,
        use_recalibrated = use_recalibrated,
        ms_level_filter  = ms_level_filter,
    )
    return c.convert()


# =============================================================================
# SECTION 7: COMMAND-LINE INTERFACE
# =============================================================================
# Allows the script to be run directly from a terminal without a Jupyter notebook:
#   python bruker_to_imzml.py /path/to/data.d --output-dir /path/to/output
#
# The argparse module parses sys.argv and maps flags to Python variables.
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a Bruker timsTOF MALDI imaging .d dataset to imzML."
    )
    parser.add_argument("d_path",
                        help="Path to .d directory")
    parser.add_argument("--output-dir",
                        help="Output directory (default: same as .d parent)")
    parser.add_argument("--output-name",
                        help="Base name for output files (no extension)")
    parser.add_argument("--sdk-path",
                        help="Explicit path to timsdata.dll / libtimsdata.so")
    parser.add_argument("--spectrum-type", default="centroid",
                        choices=["centroid", "profile"],
                        help="centroid (default) or profile")
    parser.add_argument("--no-recalibration", action="store_true",
                        help="Use raw acquisition calibration instead of post-processing recalibration")
    parser.add_argument("--ms-level-filter", type=int, default=0,
                        help="MsMsType filter: 0=MS1 (default), 2=MS2, -1=export all levels")

    args = parser.parse_args()

    # -1 is our sentinel for "no filter"; translate to None for the API
    ms_filter = None if args.ms_level_filter == -1 else args.ms_level_filter

    convert(
        bruker_d_path    = args.d_path,
        output_dir       = args.output_dir,
        output_name      = args.output_name,
        sdk_path         = args.sdk_path,
        spectrum_type    = args.spectrum_type,
        use_recalibrated = not args.no_recalibration,
        ms_level_filter  = ms_filter,
    )
