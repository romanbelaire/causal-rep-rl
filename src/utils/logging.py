"""
CSV-based logging infrastructure for metrics.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict


class CSVLogger:
    """
    CSV logger for experiment metrics.
    """
    
    def __init__(self, log_dir: str | Path, experiment_name: str, clear_existing: bool = True):
        """
        Initialize CSV logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
            clear_existing: If True, delete existing CSV file for this experiment to start fresh
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main metrics log
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"
        self.intervention_loss_file = self.log_dir / f"{experiment_name}_intervention_loss.csv"
        
        # Clear existing CSV file if it exists (to start fresh for new experiment)
        if clear_existing and self.metrics_file.exists():
            try:
                self.metrics_file.unlink()
                print(f"Cleared existing metrics file: {self.metrics_file}")
            except (IOError, OSError, PermissionError) as e:
                print(f"Warning: Could not delete existing CSV file {self.metrics_file}: {e}")
                print("  Continuing anyway - new data will overwrite existing file")
        
        if clear_existing and self.intervention_loss_file.exists():
            self.intervention_loss_file.unlink()

        self.metrics_writer = None
        self.metrics_fieldnames = None
        self.metrics_file_handle = None
        self.intervention_writer = None
        self.intervention_fieldnames = None
        self.intervention_file_handle = None
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics to CSV.
        
        Args:
            step: Training step/iteration (must be a valid integer >= 0)
            metrics: Dictionary of metric names to values
            
        Raises:
            ValueError: If step is not a valid integer >= 0
        """
        # Validate step is a valid integer
        if not isinstance(step, (int, float)):
            raise ValueError(f"step must be a number, got {type(step)}: {step}")
        step = int(step)
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if self.metrics_file_handle is None:
            # Initialize CSV writer
            self.metrics_fieldnames = ["step"] + sorted(metrics.keys())
            try:
                # Check if file exists and is corrupted - if so, remove it
                if self.metrics_file.exists():
                    try:
                        # Try to read first few bytes to check for NUL bytes or corruption
                        with open(self.metrics_file, 'rb') as f:
                            first_bytes = f.read(1024)
                            if b'\x00' in first_bytes:
                                # File contains NUL bytes - likely corrupted
                                print(f"Warning: CSV file contains NUL bytes, removing corrupted file")
                                self.metrics_file.unlink()
                    except Exception:
                        # If we can't check, try to proceed anyway
                        pass
                
                self.metrics_file_handle = open(self.metrics_file, 'w', newline='', encoding='utf-8')
                self.metrics_writer = csv.DictWriter(self.metrics_file_handle, fieldnames=self.metrics_fieldnames)
                self.metrics_writer.writeheader()
            except (IOError, OSError, PermissionError) as e:
                print(f"Error: Could not create CSV file ({e})")
                raise
        else:
            # Check if new fields need to be added
            new_fields = set(metrics.keys()) - set(self.metrics_fieldnames)
            if new_fields:
                # Add new fields to fieldnames (maintain sorted order)
                self.metrics_fieldnames = ["step"] + sorted(set(self.metrics_fieldnames[1:]) | new_fields)
                # Close current file handle before reading
                if self.metrics_file_handle is not None:
                    self.metrics_file_handle.close()
                    self.metrics_file_handle = None
                
                # Read existing rows from file (handle potential corruption/empty file)
                existing_rows = []
                try:
                    # Check if file exists and has content
                    if self.metrics_file.exists() and self.metrics_file.stat().st_size > 0:
                        with open(self.metrics_file, 'r', newline='', encoding='utf-8', errors='replace') as f:
                            # Try to read as CSV
                            try:
                                reader = csv.DictReader(f)
                                # Validate that reader has fieldnames (file has valid header)
                                if reader.fieldnames:
                                    # Read all rows, filtering out empty ones
                                    for row in reader:
                                        if row and any(v.strip() for v in row.values() if v):  # Non-empty row
                                            existing_rows.append(row)
                            except (csv.Error, ValueError, UnicodeDecodeError) as e:
                                # CSV parsing failed - file might be corrupted
                                print(f"Warning: Could not parse CSV file ({e}), starting with new header")
                                existing_rows = []
                except (FileNotFoundError, IOError, OSError, PermissionError) as e:
                    # File doesn't exist, can't be read, or permission denied - start fresh
                    print(f"Warning: Could not read existing CSV file ({e}), starting with new header")
                    existing_rows = []
                except Exception as e:
                    # Catch any other unexpected errors
                    print(f"Warning: Unexpected error reading CSV file ({e}), starting with new header")
                    existing_rows = []
                
                # Rewrite file with new header
                try:
                    self.metrics_file_handle = open(self.metrics_file, 'w', newline='', encoding='utf-8')
                    self.metrics_writer = csv.DictWriter(self.metrics_file_handle, fieldnames=self.metrics_fieldnames)
                    self.metrics_writer.writeheader()
                    # Write existing rows back (only include fields that exist in new header)
                    for row in existing_rows:
                        # Filter row to only include fields in new header, and ensure all values are strings
                        filtered_row = {}
                        for k in self.metrics_fieldnames:
                            if k in row:
                                # Convert value to string, handling None
                                val = row[k]
                                filtered_row[k] = str(val) if val is not None else ""
                            else:
                                filtered_row[k] = ""
                        try:
                            self.metrics_writer.writerow(filtered_row)
                        except Exception as e:
                            # Skip rows that can't be written
                            print(f"Warning: Could not write row to CSV ({e}), skipping")
                            continue
                except (IOError, OSError, PermissionError) as e:
                    # Can't write to file - this is a serious error
                    print(f"Error: Could not write to CSV file ({e})")
                    raise
        
        # Sanitize metrics: convert NaN/inf to strings to prevent CSV corruption
        sanitized_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                if math.isfinite(v):
                    sanitized_metrics[k] = v
                else:
                    # Replace NaN/inf with empty string (will be written as empty cell)
                    sanitized_metrics[k] = ""
            else:
                sanitized_metrics[k] = v
        
        # Write row with step as first column
        # IMPORTANT: The 'step' column should always be used as the x-axis when plotting,
        # not the CSV row index, to ensure proper alignment across training and evaluation metrics
        row = {"step": step, **sanitized_metrics}
        self.metrics_writer.writerow(row)
        self.metrics_file_handle.flush()
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_intervention_loss(self, step: int, metrics: Dict[str, float]):
        """Log v3 κ / Z* distillation training losses (sparse columns per config)."""
        if not metrics:
            return
        if not isinstance(step, (int, float)):
            raise ValueError(f"step must be a number, got {type(step)}: {step}")
        step = int(step)
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        if self.intervention_file_handle is None:
            self.intervention_fieldnames = ["step"] + sorted(metrics.keys())
            self.intervention_file_handle = open(
                self.intervention_loss_file, "w", newline="", encoding="utf-8"
            )
            self.intervention_writer = csv.DictWriter(
                self.intervention_file_handle, fieldnames=self.intervention_fieldnames
            )
            self.intervention_writer.writeheader()
        else:
            new_fields = set(metrics.keys()) - set(self.intervention_fieldnames)
            if new_fields:
                self.intervention_fieldnames = ["step"] + sorted(
                    set(self.intervention_fieldnames[1:]) | new_fields
                )
                self.intervention_file_handle.close()
                existing_rows = []
                with open(self.intervention_loss_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        existing_rows = list(reader)
                self.intervention_file_handle = open(
                    self.intervention_loss_file, "w", newline="", encoding="utf-8"
                )
                self.intervention_writer = csv.DictWriter(
                    self.intervention_file_handle, fieldnames=self.intervention_fieldnames
                )
                self.intervention_writer.writeheader()
                for row in existing_rows:
                    filtered = {k: row.get(k, "") for k in self.intervention_fieldnames}
                    self.intervention_writer.writerow(filtered)

        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                sanitized[k] = v
            else:
                sanitized[k] = ""
        row = {"step": step, **sanitized}
        self.intervention_writer.writerow(row)
        self.intervention_file_handle.flush()

    def close(self):
        """Close log files."""
        if self.metrics_file_handle is not None:
            self.metrics_file_handle.close()
        if self.intervention_file_handle is not None:
            self.intervention_file_handle.close()

