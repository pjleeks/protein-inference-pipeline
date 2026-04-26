import os
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "mmseqs2": {
        "bin": "deps/mmseqs2/build/bin/mmseqs",
        "default_gpu": 0,
    },
    "protenix": {
        "runner": "deps/protenix/runner/inference.py",
        "default_gpu": 1,
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_dir: Path) -> logging.Logger:
    """Setup dual-output logging (file + console) with timestamps."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (detailed)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simpler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_dependencies(root: Path, logger: logging.Logger) -> None:
    """Validate that MMseqs2 and Protenix binaries exist."""
    mmseqs_bin = root / MODEL_CONFIG["mmseqs2"]["bin"]
    protenix_runner = root / MODEL_CONFIG["protenix"]["runner"]
    
    if not mmseqs_bin.exists():
        raise FileNotFoundError(
            f"MMseqs2 binary not found: {mmseqs_bin}\n"
            f"Please ensure MMseqs2 is compiled in deps/mmseqs2/build/bin/"
        )
    
    if not protenix_runner.exists():
        raise FileNotFoundError(
            f"Protenix runner not found: {protenix_runner}\n"
            f"Please ensure Protenix is available in deps/protenix/runner/"
        )
    
    logger.info("✓ All dependencies validated")


def validate_fasta(fasta_path: Path, logger: logging.Logger) -> tuple[str, str]:
    """
    Validate FASTA file and extract sequence ID.
    
    Returns:
        tuple: (sequence_id, sequence)
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    try:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError("FASTA file is empty")
        
        if not lines[0].startswith(">"):
            raise ValueError("FASTA file does not start with header (>)")
        
        # Extract sequence ID from header
        seq_id = lines[0].strip().lstrip(">").split()[0]
        
        # Extract sequence
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
        
        if not sequence:
            raise ValueError("No sequence found in FASTA file")
        
        logger.info(f"✓ FASTA validated: {len(sequence)} bp, ID: {seq_id}")
        return seq_id, sequence
        
    except Exception as e:
        raise ValueError(f"Error parsing FASTA file {fasta_path}: {str(e)}")


def validate_database(db_path: Path, logger: logging.Logger) -> None:
    """Validate MMseqs2 database exists."""
    if not db_path.exists():
        raise FileNotFoundError(f"MMseqs2 database not found: {db_path}")
    
    logger.info(f"✓ Database validated: {db_path}")


# ============================================================================
# GPU MANAGEMENT
# ============================================================================

def get_available_gpus(logger: logging.Logger) -> List[int]:
    """Detect available GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            "nvidia-smi --list-gpus",
            shell=True, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            gpu_count = len([g for g in gpus if g.strip()])
            logger.info(f"✓ Detected {gpu_count} GPU(s)")
            return list(range(gpu_count))
    except Exception as e:
        logger.warning(f"Could not detect GPUs: {e}")
    
    return []


def assign_gpus(logger: logging.Logger) -> Dict[str, int]:
    """
    Assign GPUs to models. If <2 GPUs available, use GPU 0 for both.
    
    Returns:
        dict: {"mmseqs2": gpu_id, "protenix": gpu_id}
    """
    available_gpus = get_available_gpus(logger)
    
    if len(available_gpus) >= 2:
        assignment = {
            "mmseqs2": MODEL_CONFIG["mmseqs2"]["default_gpu"],
            "protenix": MODEL_CONFIG["protenix"]["default_gpu"]
        }
        logger.info(f"Using separate GPUs: MMseqs2→GPU{assignment['mmseqs2']}, Protenix→GPU{assignment['protenix']}")
    else:
        assignment = {
            "mmseqs2": 0,
            "protenix": 0
        }
        logger.warning("Only 1 GPU available. Running both models sequentially on GPU 0")
    
    return assignment


# ============================================================================
# COMMAND EXECUTION
# ============================================================================

def run_command(command: str, description: str, logger: logging.Logger, timeout: int = None) -> str:
    """
    Execute a shell command with comprehensive error logging.
    
    Args:
        command: Shell command to execute
        description: Human-readable description of the operation
        logger: Logger instance
        timeout: Optional timeout in seconds
        
    Returns:
        str: stdout output
        
    Raises:
        Exception: If command fails
    """
    logger.info(f"{'='*70}")
    logger.info(f"Starting: {description}")
    logger.info(f"{'='*70}")
    logger.debug(f"Command: {command}")
    
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"Command timed out after {timeout} seconds")
            raise
        
        # Log output
        if stdout:
            logger.debug(f"STDOUT:\n{stdout}")
        if stderr:
            logger.warning(f"STDERR:\n{stderr}")
        
        if process.returncode != 0:
            logger.error(f"✗ {description} FAILED (return code: {process.returncode})")
            logger.error(f"STDERR: {stderr}")
            raise Exception(
                f"{description} failed with return code {process.returncode}\n"
                f"Error: {stderr}"
            )
        
        logger.info(f"✓ {description} completed successfully")
        return stdout
        
    except Exception as e:
        logger.error(f"✗ Exception during {description}: {str(e)}")
        raise


# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def save_checkpoint(checkpoint_file: Path, data: Dict, logger: logging.Logger) -> None:
    """Save pipeline checkpoint for resumption."""
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_file: Path, logger: logging.Logger) -> Optional[Dict]:
    """Load pipeline checkpoint if it exists."""
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint: {checkpoint_file}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_protenix_json(
    fasta_path: Path,
    msa_path: Path,
    output_json: Path,
    logger: logging.Logger
) -> None:
    """
    Create Protenix-compatible JSON from FASTA and MSA files.
    
    Includes metadata tracking for inter-model communication.
    """
    # Parse FASTA
    seq_id, sequence = validate_fasta(fasta_path, logger)
    
    # Validate MSA
    if not msa_path.exists():
        raise FileNotFoundError(f"MSA file not found: {msa_path}")
    
    msa_size = msa_path.stat().st_size
    logger.info(f"MSA file: {msa_path} ({msa_size:,} bytes)")
    
    # Structure required by Protenix (AF3-style)
    input_data = [{
        "name": seq_id or fasta_path.stem,
        "sequences": [
            {
                "protein": {
                    "sequence": sequence,
                    "unpaired_msa_path": str(msa_path.absolute()),
                }
            }
        ],
        "model_seeds": [101],
        # Metadata for tracking and debugging
        "metadata": {
            "source_fasta": str(fasta_path.absolute()),
            "sequence_id": seq_id,
            "sequence_length": len(sequence),
            "msa_size_bytes": msa_size,
            "generated_at": datetime.now().isoformat(),
        }
    }]

    with open(output_json, 'w') as f:
        json.dump(input_data, f, indent=4)
    
    logger.info(f"✓ Created Protenix input JSON: {output_json}")


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def run_mmseqs2(
    fasta_path: Path,
    db_path: Path,
    msa_out: Path,
    gpu_id: int,
    logger: logging.Logger,
    skip: bool = False
) -> None:
    """Run MMseqs2 MSA generation."""
    if skip and msa_out.exists():
        logger.info(f"Skipping MMseqs2: MSA file already exists at {msa_out}")
        return
    
    root = Path(__file__).parent.absolute()
    mmseqs_bin = root / MODEL_CONFIG["mmseqs2"]["bin"]
    
    mmseqs_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} {mmseqs_bin} easy-search "
        f"{fasta_path} {db_path} {msa_out} tmp "
        f"--format-mode 4"  # Mode 4 produces A3M
    )
    
    run_command(mmseqs_cmd, "MMseqs2 MSA Generation", logger)


def run_protenix(
    protenix_json: Path,
    out_dir: Path,
    gpu_id: int,
    logger: logging.Logger
) -> None:
    """Run Protenix inference."""
    root = Path(__file__).parent.absolute()
    protenix_runner = root / MODEL_CONFIG["protenix"]["runner"]
    
    protenix_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {protenix_runner} "
        f"--input {protenix_json} "
        f"--output_dir {out_dir}"
    )
    
    run_command(protenix_cmd, "Protenix Inference", logger)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Protein Inference Pipeline: MMseqs2 + Protenix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference.py --fasta protein.fasta --db /path/to/db

  # Skip MSA generation if already computed
  python inference.py --fasta protein.fasta --db /path/to/db --skip_mmseqs2

  # Resume from checkpoint after interruption
  python inference.py --fasta protein.fasta --db /path/to/db --resume
        """
    )
    
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to input .fasta file"
    )
    parser.add_argument(
        "--db", type=str, required=True,
        help="Path to MMseqs2 database"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results",
        help="Directory for outputs (default: ./results)"
    )
    parser.add_argument(
        "--skip_mmseqs2", action="store_true",
        help="Skip MSA generation if file already exists"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    root = Path(__file__).parent.absolute()
    fasta_path = Path(args.fasta).resolve()
    db_path = Path(args.db).resolve()
    out_path = Path(args.out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(out_path)
    
    try:
        logger.info("="*70)
        logger.info("PROTEIN INFERENCE PIPELINE: MMseqs2 + Protenix")
        logger.info("="*70)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"FASTA: {fasta_path}")
        logger.info(f"Database: {db_path}")
        logger.info(f"Output: {out_path}")
        logger.info("="*70)
        
        # Checkpoint paths
        checkpoint_file = out_path / "pipeline_checkpoint.json"
        msa_out = out_path / "msa.a3m"
        protenix_json = out_path / "protenix_input.json"
        
        # Load or initialize checkpoint
        checkpoint = load_checkpoint(checkpoint_file, logger) if args.resume else None
        
        # Validate dependencies
        validate_dependencies(root, logger)
        validate_database(db_path, logger)
        validate_fasta(fasta_path, logger)
        
        # Assign GPUs
        gpu_assignment = assign_gpus(logger)
        
        # ===== STAGE 1: MMseqs2 MSA Generation =====
        if checkpoint and checkpoint.get("mmseqs2_complete"):
            logger.info("Resuming: Skipping MMseqs2 (already completed)")
        else:
            logger.info("\n[STAGE 1/3] MMseqs2 MSA Generation")
            run_mmseqs2(
                fasta_path, db_path, msa_out,
                gpu_assignment["mmseqs2"],
                logger,
                skip=args.skip_mmseqs2
            )
            save_checkpoint(
                checkpoint_file,
                {"mmseqs2_complete": True, "msa_path": str(msa_out)},
                logger
            )
        
        # ===== STAGE 2: Data Bridge =====
        if checkpoint and checkpoint.get("protenix_json_complete"):
            logger.info("Resuming: Skipping JSON creation (already completed)")
        else:
            logger.info("\n[STAGE 2/3] Creating Protenix Input JSON")
            create_protenix_json(fasta_path, msa_out, protenix_json, logger)
            save_checkpoint(
                checkpoint_file,
                {"mmseqs2_complete": True, "protenix_json_complete": True},
                logger
            )
        
        # ===== STAGE 3: Protenix Inference =====
        if checkpoint and checkpoint.get("protenix_complete"):
            logger.info("Resuming: Protenix already completed")
        else:
            logger.info("\n[STAGE 3/3] Protenix Inference")
            run_protenix(
                protenix_json, out_path,
                gpu_assignment["protenix"],
                logger
            )
            save_checkpoint(
                checkpoint_file,
                {
                    "mmseqs2_complete": True,
                    "protenix_json_complete": True,
                    "protenix_complete": True,
                    "completed_at": datetime.now().isoformat()
                },
                logger
            )
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Results directory: {out_path}")
        logger.info(f"Log file: {out_path}/pipeline_*.log")
        logger.info("="*70)
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("✗ PIPELINE FAILED!")
        logger.error("="*70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("="*70)
        logger.error(f"Use --resume flag to retry from checkpoint")
        raise


if __name__ == "__main__":
    main()
