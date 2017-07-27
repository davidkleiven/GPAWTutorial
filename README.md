# Errors
When running the scripts GPAW was giving a segmentation fault.
This was found to be caused by FFTW3
```bash
export GPAW_FFTWSO=''
```
fixed the issue.

If an error that GPAW cannot find the PAW files
```bash
export GPAW_SETUP_PATH=/path/to/gpaw/datasets
```
solves the problem.
