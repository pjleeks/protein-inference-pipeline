import os
import json
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    print(f"--- Starting {description} ---")
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise Exception(f"{description} failed with return code {process.returncode}")

def create_protenix_json(fasta_path, msa_path, output_json):
    """Links MMseqs2 output to Protenix-compatible JSON."""
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
        # Extract sequence, ignoring header lines
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])

    # Structure required by Protenix (AF3-style)
    input_data = [{
        "name": Path(fasta_path).stem,
        "sequences": [
            {
                "protein": {
                    "sequence": sequence,
                    "unpaired_msa_path": str(Path(msa_path).absolute()),
                }
            }
        ],
        "model_seeds": [101]
    }]

    with open(output_json, 'w') as f:
        json.dump(input_data, f, indent=4)
    print(f"Created Protenix input JSON: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Protein Inference Pipeline: MMseqs2 + Protenix")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input .fasta file")
    parser.add_argument("--db", type=str, required=True, help="Path to MMseqs2 database")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory for final outputs")
    args = parser.parse_args()

    # Setup Paths
    root = Path(__file__).parent.absolute()
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    msa_out = out_path / "msa.a3m"
    protenix_json = out_path / "protenix_input.json"
    
    # 1. RUN MMseqs2 (GPU 0)
    # We use the easy-search workflow to get an A3M file
    # Ensure you've compiled MMseqs2 in deps/mmseqs2/build/bin/mmseqs
    mmseqs_bin = root / "deps/mmseqs2/build/bin/mmseqs"
    mmseqs_cmd = (
        f"CUDA_VISIBLE_DEVICES=0 {mmseqs_bin} easy-search "
        f"{args.fasta} {args.db} {msa_out} tmp "
        f"--format-mode 4" # Mode 4 produces A3M
    )
    run_command(mmseqs_cmd, "MMseqs2 MSA Generation")

    # 2. CREATE THE LINK (Bridge Logic)
    create_protenix_json(args.fasta, msa_out, protenix_json)

    # 3. RUN PROTENIX (GPU 1)
    # We call the runner script from the submodule
    protenix_runner = root / "deps/protenix/runner/inference.py"
    protenix_cmd = (
        f"CUDA_VISIBLE_DEVICES=1 python {protenix_runner} "
        f"--input {protenix_json} "
        f"--output_dir {out_path}"
    )
    run_command(protenix_cmd, "Protenix Inference")

    print(f"\nPipeline Complete! Results are in: {args.out_dir}")

if __name__ == "__main__":
    main()
