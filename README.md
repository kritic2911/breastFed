# fl-mammo: week-1 data pipeline

Turns each "hospital" dataset into a small, uniform cache (512px PNGs + a labels CSV) with
patient-level splits. Raw data is never needed again after this step.

## Setup (inside WSL, project in ~/fl-mammo)

    python3 -m venv .venv && source ../venv/bin/activate
    pip install -r requirements.txt
    # then install torch/torchvision from pytorch.org and check torch.cuda.is_available()

Put these scripts in `scripts/`. All commands run from the project root. Output goes to `data/<site>/`.

## CMMD (DICOM)

    python scripts/ingest_dicom.py --dicom-dir /path/to/CMMD --site cmmd
    python scripts/build_labels_cmmd.py --clinical /path/to/CMMD_clinicaldata_revision.xlsx
    python scripts/make_splits.py --site cmmd
    python scripts/preview.py --site cmmd

Add `--delete` to ingest_dicom.py to remove each DICOM once it is cached (stream-and-delete).

## INbreast / VinDr (category-sorted PNGs)

    python scripts/ingest_png.py --root /path/to/inbreast_pngs --site inbreast --list
    python scripts/ingest_png.py --root /path/to/inbreast_pngs --site inbreast \
        --map "folder_a=0,folder_b=1,folder_c=drop" --patient-regex "^(\d+)_"
    python scripts/make_splits.py --site inbreast
    python scripts/preview.py --site inbreast

Use `--list` first to see folder names and sample filenames, then write the map and the regex.

## Two things to decide before trusting any result

1. **Patient grouping.** Splits must be by patient. The PNG sites only get this if the filename
   contains a patient/study ID that `--patient-regex` can extract. If it does not, the scripts warn
   loudly and fall back to image-level splits, which leak.
2. **Label meaning differs by site.** CMMD labels are biopsy-confirmed benign/malignant and contain
   no normal cases. INbreast and VinDr categories are BI-RADS radiologist assessments. Dropping the
   BI-RADS 1 (normal) folders and mapping 2-3 -> 0, 4-5 -> 1 keeps the task closest to CMMD's, but it
   is still not the same label. Write the mapping you used into your report.

Open `data/<site>/preview.png` for every site and check the crops before training anything.

---

## Training Pipeline & Benchmarking

See the full **[Usage Guide (USAGE_GUIDE.md)](USAGE_GUIDE.md)** for execution commands, hardware optimizations (in-memory caching & batch processing), and artifact storage.

- **Local-only baseline**: `python scripts/run_local.py --seed 42` (or `--all-seeds`)
- **Centralized baseline**: `python scripts/run_centralized.py --site INbreast --seed 42` (or `--site CMMD`)
- **Federated (FedAvg)**: `python scripts/run_federated.py --seed 42`
- **Cross-site gap (LOSO)**: `python scripts/run_cross_site.py --seed 42`
- **Experiment Log**: Real-time run logs and metrics table in `experiment_log.md`

