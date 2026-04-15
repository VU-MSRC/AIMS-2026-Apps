# AIMS-2026-Apps
Data analysis tutorial workshop apps provided by ME Colley for the 2026 annual advanced imaging mass spectrometry workshop at Vanderbilt University MSRC. 

## CONDAEnvironment
This folder contains the .yaml file that you can use to make sure your anaconda environment is equipped with the necessary plugins and versions to run 2 apps contained in this repo. Make a new environment and import environment.yml to build a clone of the one I used to make these apps. 

## BrukerTSFtoimzMLConverter
This small app is a python program that reads the Bruker tsf (qTOF/not TIMS) format natively and generates an imzmL file (both ibd and imzml components) that can be used in other downstream analysis with programs such as Cardinal MSI. The .dlls were provided by Bruker and retrieved from the tdf-sdk downloaded from bruker.com. 

Load the provided juypter notebook and copy the data paths do your data path in Cell 3
```# Full path to your Bruker .d directory
BRUKER_D_PATH = r"C:\YOUR\DATA\PATH\HERE\datafile.d"

# Where to write the output .imzML and .ibd files
OUTPUT_DIR = r"C:\YOUR\DATA\PATH\HERE\outputs"

# Base name for output files (no extension)
# → OUTPUT_DIR/OUTPUT_NAME.imzML  and  OUTPUT_DIR/OUTPUT_NAME.ibd
OUTPUT_NAME = 'NAMEYOURIMZMLHERE'

# ── SDK PATH ────────────────────────────────────────────────────────────────

# Full path to timsdata.dll (Windows) or libtimsdata.so (Linux).
# Set to None if the library is already on PATH / LD_LIBRARY_PATH (Option B).

SDK_PATH = r'C:\PATHTODLLHERE\timsdata.dll'
# SDK_PATH = '/path/to/timsdata/linux64/libtimsdata.so'      # Linux example```
