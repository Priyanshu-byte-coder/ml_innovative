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
    parser.add_argument('--max_samples', type=int, default=200000,
                        help="Number of reviews to sample (default 200000)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sim_threshold', type=float, default=0.8,
                        help="Cosine similarity threshold for text edges")
    parser.add_argument('--fake_threshold', type=float, default=0.35,
                        help="Decision threshold for fake classification")
    parser.add_argument('--balanced_sampling', type=str, default='True',
                        help="Use class-balanced mini-batch sampling")
    parser.add_argument('--hard_example_mining', type=str, default='True',
                        help="Use hard example mining during training")
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
             '--max_samples', str(args.max_samples),
             '--sim_threshold', str(args.sim_threshold)])

        if not args.skip_train:
            # Step 2: Train
            run([sys.executable, 'src/train.py',
                 '--data_dir', 'processed',
                 '--output_dir', 'checkpoints',
                 '--epochs', str(args.epochs),
                 '--fake_threshold', str(args.fake_threshold),
                 '--balanced_sampling', args.balanced_sampling,
                 '--hard_example_mining', args.hard_example_mining])

    # Step 3: Launch web app
    print("\n" + "="*60)
    print("  Launching web app at http://localhost:5000")
    print("="*60 + "\n")
    run([sys.executable, 'app/app.py'])


if __name__ == '__main__':
    main()
