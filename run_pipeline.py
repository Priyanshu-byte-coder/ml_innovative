"""
Run the full pipeline: preprocess → train → launch web app.
Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --skip_train     # preprocess only, then launch app
    python run_pipeline.py --app_only       # just launch the web app
"""
import subprocess
import sys
import os
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=ROOT):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=50000,
                        help="Number of reviews to sample (default 50000)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--skip_train', action='store_true',
                        help="Skip training, just preprocess + app")
    parser.add_argument('--app_only', action='store_true',
                        help="Skip preprocess + train, just launch app")
    args = parser.parse_args()

    if not args.app_only:
        # Step 1: Preprocess
        run([sys.executable, 'src/preprocess.py',
             '--data_dir', 'Yelp-Dataset',
             '--output_dir', 'processed',
             '--max_samples', str(args.max_samples)])

        if not args.skip_train:
            # Step 2: Train
            run([sys.executable, 'src/train.py',
                 '--data_dir', 'processed',
                 '--output_dir', 'checkpoints',
                 '--epochs', str(args.epochs)])

    # Step 3: Launch web app
    print("\n" + "="*60)
    print("  Launching web app at http://localhost:5000")
    print("="*60 + "\n")
    run([sys.executable, 'app/app.py'])


if __name__ == '__main__':
    main()
