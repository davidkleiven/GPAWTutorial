# Errors
When running the scripts GPAW was giving a segmentation fault.
This was found to be caused by FFTW3
```bash
export GPAW_FFTWSO=''
```
fixed the issue.
