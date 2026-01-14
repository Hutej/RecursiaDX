# CSV Deduplication System

## Overview

The `deduplicate_csvs.py` utility automatically removes duplicate slide entries from label CSV files while maintaining timestamped backups.

---

## Features

- ✅ **Automatic Backup**: Creates timestamped backup before any changes
- ✅ **Smart Deduplication**: Keeps only the latest entry per `slide_name`
- ✅ **Dry-Run Mode**: Preview changes before executing
- ✅ **Safe Design**: Never touches tile or feature files
- ✅ **Comprehensive Logging**: Clear summary of all changes

---

## Usage

### Preview Duplicates (Recommended First Step)

```bash
python scripts/deduplicate_csvs.py --dry-run
```

**Output**:
```
DRY-RUN MODE: Previewing changes (no files will be modified)
Found 4 CSV files to process

Processing: labels.csv
labels.csv: Found 3 duplicate entries
  Duplicate slides: slide_001, slide_003, slide_005
labels.csv: Removed 3 duplicates
  Before: 203 rows → After: 200 rows

SUMMARY
Total slides scanned: 203
Total duplicates found: 3
⚠️  No changes made (dry-run mode)
Run with --execute to apply changes
```

### Execute Deduplication

```bash
python scripts/deduplicate_csvs.py --execute
```

**Output**:
```
Creating backup in: data/backup_labels_20260112_074847

Processing: labels.csv
labels.csv: Removed 3 duplicates
✓ Saved cleaned labels.csv

SUMMARY
Total duplicates found: 3
Backup location: data/backup_labels_20260112_074847
✅ Deduplication complete!
```

### Custom Data Directory

```bash
python scripts/deduplicate_csvs.py --data-dir custom_data/ --execute
```

---

## How It Works

### 1. Backup Creation

Before any changes, creates timestamped backup:

```
data/backup_labels_20260112_074847/
├── labels.csv
├── labels_train.csv
├── labels_val.csv
└── labels_test.csv
```

### 2. Duplicate Detection

Scans for duplicate `slide_name` entries across:
- `labels.csv`
- `labels_train.csv`
- `labels_val.csv`
- `labels_test.csv`

### 3. Keep Latest Entry

Uses `pandas.drop_duplicates(subset=['slide_name'], keep='last')`:
- Preserves the **most recent** entry
- Removes all older duplicates

### 4. Permanent Deletion

Duplicates are **permanently removed** (no archiving or inactive rows).

---

## Safety Guarantees

### What Is Modified

✅ CSV files only:
- `data/labels.csv`
- `data/labels_train.csv`
- `data/labels_val.csv`
- `data/labels_test.csv`

### What Is NEVER Touched

❌ **Tiles** (`data/tiles/`)  
❌ **Features** (`data/features/`)  
❌ **Top-K Features** (`data/features_topk/`)  
❌ **Models** (`checkpoints/`)  
❌ **Results** (`results/`)

---

## Integration with Pipeline

### Recommended Workflow

```bash
# 1. Check for duplicates after label generation
python scripts/regenerate_labels.py
python scripts/deduplicate_csvs.py --dry-run

# 2. If duplicates found, execute deduplication
python scripts/deduplicate_csvs.py --execute

# 3. Proceed with training
python scripts/train_mil.py --features data/features_topk --labels data/labels.csv
```

### Automatic Integration (Optional)

Add to `regenerate_labels.py` or similar scripts:

```python
import subprocess

# After generating labels
subprocess.run(['python', 'scripts/deduplicate_csvs.py', '--execute'], check=True)
```

---

## Exit Codes

- `0`: Success (no duplicates or cleaned successfully)
- `1`: Duplicates found in dry-run mode OR error occurred

---

## Logging

Logs saved to: `logs/csv_deduplication_YYYYMMDD_HHMMSS.log`

**Log Contents**:
- Files scanned
- Duplicates detected
- Rows removed
- Backup location
- Per-file statistics

---

## Recovery from Backup

If deduplication causes issues:

```bash
# List backups
ls -altr data/backup_labels_*

# Restore from backup
cp data/backup_labels_20260112_074847/* data/
```

---

## Example Scenarios

### Scenario 1: Clean CSVs

```bash
$ python scripts/deduplicate_csvs.py --dry-run

Processing: labels.csv
labels.csv: No duplicates found (200 unique rows)

✓ All CSVs are clean (no duplicates)
```

**Action**: None needed.

---

### Scenario 2: Found Duplicates

```bash
$ python scripts/deduplicate_csvs.py --dry-run

labels.csv: Found 5 duplicate entries
labels_train.csv: Found 3 duplicate entries

Total duplicates found: 8
⚠️  Run with --execute to remove them

$ python scripts/deduplicate_csvs.py --execute

Backup created: data/backup_labels_20260112_075030
✓ Removed 8 duplicates
```

**Action**: Duplicates permanently removed, backup available.

---

## Advanced Usage

### Check Specific Directory

```bash
python scripts/deduplicate_csvs.py --data-dir experiments/run_01/ --dry-run
```

### Automated Script

```bash
#!/bin/bash
# auto_deduplicate.sh

echo "Checking for duplicates..."
if python scripts/deduplicate_csvs.py --dry-run; then
    echo "No duplicates found"
else
    echo "Duplicates detected, cleaning..."
    python scripts/deduplicate_csvs.py --execute
fi
```

---

## Troubleshooting

### Issue: "No 'slide_name' column found"

**Cause**: CSV missing required column.

**Solution**: Ensure CSV has `slide_name` column:

```csv
slide_name,label
slide_001,0
slide_002,1
```

### Issue: Backup failed

**Cause**: Insufficient disk space or permissions.

**Solution**: Check disk space and write permissions:

```bash
df -h data/
ls -ld data/
```

---

## Best Practices

1. **Always Dry-Run First**: Preview changes before executing
2. **Keep Backups**: Don't delete backup folders immediately
3. **Run Periodically**: After label regeneration or updates
4. **Verify Results**: Check CSV row counts after deduplication

---

**Last Updated**: 2026-01-12

**Script**: `scripts/deduplicate_csvs.py`
