import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zuco_inspector import ZuCoInspector
from src.utils.logger import setup_logger

logger = setup_logger("step0_inspect_data")


def main():
    parser = argparse.ArgumentParser(description="Step 0: Inspect ZuCo 1.0 and ZuCo 2.0 Dataset Structure")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing ZuCo 1.0 and ZuCo 2.0 files (.mat or .pkl)",
    )
    parser.add_argument(
        "--output_doc",
        type=str,
        default="docs/data_inspection.md",
        help="Output path for the generated markdown inspection report",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/zuco_unified_index.csv",
        help="Output path for the unified dataset index CSV",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit on number of files to inspect (for fast preview)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_doc = PROJECT_ROOT / args.output_doc if not Path(args.output_doc).is_absolute() else Path(args.output_doc)
    output_csv = PROJECT_ROOT / args.output_csv if not Path(args.output_csv).is_absolute() else Path(args.output_csv)

    logger.info("=" * 70)
    logger.info("STEP 0: MANDATORY DATA INSPECTION (ZuCo 1.0 & ZuCo 2.0)")
    logger.info(f"Target Data Directory: {data_dir.resolve()}")
    logger.info(f"Report Output Path:    {output_doc.resolve()}")
    logger.info(f"Index CSV Output Path: {output_csv.resolve()}")
    logger.info("=" * 70)

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    inspector = ZuCoInspector(data_dir)
    df = inspector.inspect_all(max_files=args.max_files)

    if df.empty:
        logger.error("No valid ZuCo records found! Please verify the folder contents and file extensions.")
        sys.exit(1)

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Successfully created unified dataset index ({len(df):,} trials) at: {output_csv}")

    # Generate Markdown Report
    report_md = inspector.generate_inspection_report(df, output_doc)
    logger.info(f"Successfully generated inspection report at: {output_doc}")

    # Print summary highlights to terminal
    print("\n" + "=" * 70)
    print("DATA INSPECTION SUMMARY HIGHLIGHTS")
    print("=" * 70)
    print(f"Total Trials Recorded:     {len(df):,}")
    print(f"Datasets:                  {list(df['dataset'].unique())}")
    print(f"Total Subjects:            {df['subject'].nunique()} ({list(df['subject'].unique())[:8]}...)")
    print(f"Tasks:                     {list(df['task'].unique())}")
    print(f"Channel Counts:            {list(df['n_channels'].unique())}")
    print(f"Sampling Rates:            {list(df['sampling_rate'].unique())} Hz")
    print(f"Mean Sentence Duration:    {df['duration_sec'].mean():.2f} seconds")
    print(f"Mean Word Count:           {df['word_count'].mean():.1f} words")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
