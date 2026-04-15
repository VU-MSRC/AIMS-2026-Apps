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
# SDK_PATH = '/path/to/timsdata/linux64/libtimsdata.so'      # Linux example
```

To run this code, simply start at Cell 1 and click the play button through each cell and wait patiently for your imzML to build. The imzML will be available in your specified outputs folder. If you run into any issues, please commit them here or email madeline.colley@vanderbilt.edu. Please do not email me about setting up jupyter or anaconda! Google will be better than I am at helping you out there.

## IonImager
IonImager is another python based jupyter notebook used to display chosen m/z values from the Bruker .d (TSF, not TIMS) directly. This app is simple on purpose and is a proof-of-concept and starting point to build code to interpret, analyze and tinker with Bruker .d format data directly and without having to convert to other open-source data formats (e.g. imzML). My hope is that this snippet is an inspiring starting point for data scientists to build open-source apps and programs that can read imaging mass spectrometry data from the native format. If you are a user of other instrument vendors (e.g. Waters, Thermo, Shimadzu), I would look forward to working with you on data you have collected to build similar apps.

Cell 4 is editable for your individual data set.
```D_FOLDER  = r"C:\YOUR\DATA\PATH\data.d"
MZ_MIN    = 885.53
MZ_MAX    = 885.55
NORMALIZE = False   # set True to divide by TIC

image = extract_ion_image(D_FOLDER, MZ_MIN, MZ_MAX, normalize=NORMALIZE)
```

