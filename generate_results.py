#!/usr/bin/env python3
"""
Generate a spreadsheet of analysis.py results for different k values and input files.
Outputs results to results.csv
"""

import subprocess
import csv
import os
from pathlib import Path

def run_analysis(k, input_file):
    """
    Run analysis.py with given k and input_file.
    Returns (nmf_score, kmeans_score) or (None, None) on error.
    """
    try:
        result = subprocess.run(
            ['python3', 'analysis.py', str(k), input_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"Warning: analysis.py failed for k={k}, file={input_file}")
            return None, None
        
        # Parse output: expect lines like "nmf: X.XXXX" and "kmeans: X.XXXX"
        lines = result.stdout.strip().split('\n')
        nmf_score = None
        kmeans_score = None
        
        for line in lines:
            if line.startswith('nmf:'):
                nmf_score = float(line.split(':')[1].strip())
            elif line.startswith('kmeans:'):
                kmeans_score = float(line.split(':')[1].strip())
        
        return nmf_score, kmeans_score
    except Exception as e:
        print(f"Error running analysis.py for k={k}, file={input_file}: {e}")
        return None, None


def main():
    # Define parameters
    
    input_files = [
        ('input_1.txt', list(range(2, 11))),
        ('input_2.txt', list(range(2, 11))),
        ('input_3.txt', list(range(2, 5))),
    ]
    
    # Verify all input files exist
    for input_file, k_values in input_files:
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found")
            return
    
    # Collect results
    results = []
    total = sum(len(k_values) for _, k_values in input_files)
    current = 0
    
    for input_file, k_values in input_files:
        for k in k_values:
            current += 1
            print(f"[{current}/{total}] Running analysis.py with k={k}, file={input_file}...", end=' ')
            nmf_score, kmeans_score = run_analysis(k, input_file)
            if nmf_score is not None:
                results.append({
                    'input_file': input_file,
                    'k': k,
                    'nmf_score': nmf_score,
                    'kmeans_score': kmeans_score,
                })
                print(f"nmf={nmf_score:.4f}, kmeans={kmeans_score:.4f}")
            else:
                print("FAILED")
    
    # Write results to CSV
    output_file = 'results.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['input_file', 'k', 'nmf_score', 'kmeans_score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults written to {output_file}")
    print(f"Total rows: {len(results)}")
    
    # Also print a summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    for input_file in input_files:
        print(f"\n{input_file}:")
        print(f"{'k':<5} {'NMF Score':<15} {'KMeans Score':<15}")
        print("-" * 35)
        for row in results:
            if row['input_file'] == input_file:
                print(f"{row['k']:<5} {row['nmf_score']:<15.4f} {row['kmeans_score']:<15.4f}")


if __name__ == '__main__':
    main()
