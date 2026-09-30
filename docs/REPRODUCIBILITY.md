# Reproducibility Guide

## What Is Included

The repository includes the code paths for:

- converting authorized clips into audio and MediaPipe skeleton features;
- normalizing the prepared dataset;
- training the six-class classifier;
- grouped and stratified evaluation;
- paper figures and unit tests.

## What Is Intentionally Excluded

Raw source videos, derived WAV files, skeleton CSV files, trained model
binaries, and annotation media are ignored. The research assets originated from
online performance and teaching sources and need copyright and permission review
before any separate release.

## Local Setup

Use a dedicated virtual environment or Anaconda environment outside the system
drive when possible:

```powershell
D:\Anaconda\python.exe -m pip install -r requirements.txt
```

Place only authorized local data into the ignored technique folders and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce.ps1
```

The test suite can run without the raw dataset:

```powershell
D:\Anaconda\python.exe -m pytest tests
```

## Reporting

The reported metrics are for an internal research prototype with a small,
author-labelled dataset. Present the project as multimodal technique
recognition, not a production grading system or a public benchmark.
