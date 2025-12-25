#!/usr/bin/env python
"""
Run all git-tracked Jupyter notebooks and check for errors.

Usage:
    python scripts/run_notebooks.py          # Run all tracked notebooks
    python scripts/run_notebooks.py --clear  # Clear outputs before running
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def get_tracked_notebooks(project_root: Path, notebooks_dir: str) -> list[Path]:
    """Get list of git-tracked notebooks."""
    try:
        result = subprocess.run(
            ["git", "ls-files", f"{notebooks_dir}/*.ipynb"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked = [project_root / line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return sorted(tracked)
    except subprocess.CalledProcessError:
        return []


def run_notebook(notebook_path: Path, clear_output: bool = False) -> bool:
    """
    Execute a notebook and return True if successful, False if any errors.
    
    Args:
        notebook_path: Path to the notebook file
        clear_output: If True, clear outputs before running
        
    Returns:
        True if notebook executed without errors, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path.name}")
    print('='*60)
    
    try:
        # Execute the notebook in place
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",  # 5 minute timeout per cell
            str(notebook_path)
        ]
        
        if clear_output:
            # First clear outputs
            clear_cmd = [
                "jupyter", "nbconvert",
                "--ClearOutputPreprocessor.enabled=True",
                "--to", "notebook",
                "--inplace",
                str(notebook_path)
            ]
            subprocess.run(clear_cmd, check=True, capture_output=True)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"{Colors.RED}FAILED:{Colors.RESET} {notebook_path.name}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print(f"{Colors.GREEN}PASSED:{Colors.RESET} {notebook_path.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}FAILED:{Colors.RESET} {notebook_path.name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"{Colors.RED}ERROR:{Colors.RESET} jupyter not found. Install with: pip install jupyter nbconvert")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all git-tracked Jupyter notebooks and check for errors")
    parser.add_argument("--clear", action="store_true", help="Clear outputs before running")
    parser.add_argument("--notebooks-dir", type=str, default="notebooks", help="Directory containing notebooks")
    args = parser.parse_args()
    
    # Find project root (where pyproject.toml is)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Get only git-tracked notebooks
    notebooks = get_tracked_notebooks(project_root, args.notebooks_dir)
    
    if not notebooks:
        print(f"No git-tracked notebooks found in {args.notebooks_dir}/")
        sys.exit(0)
    
    print(f"Found {len(notebooks)} git-tracked notebook(s) to run:")
    for nb in notebooks:
        print(f"  - {nb.name}")
    
    # Run all notebooks
    results = {}
    for notebook in notebooks:
        results[notebook.name] = run_notebook(notebook, clear_output=args.clear)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passed = sum(1 for r in results.values() if r)
    failed = sum(1 for r in results.values() if not r)
    
    for name, success in results.items():
        if success:
            status = f"{Colors.GREEN}PASSED{Colors.RESET}"
        else:
            status = f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  [{status}] {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    
    print(f"\n{Colors.GREEN}All notebooks executed successfully!{Colors.RESET}")
    sys.exit(0)


if __name__ == "__main__":
    main()
