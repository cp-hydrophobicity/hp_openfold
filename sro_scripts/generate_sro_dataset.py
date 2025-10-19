from openfold.model.sro.data import build_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate SRO dataset")
    parser.add_argument("--predictions_dir", type=str, required=True,
                        help="Path to the predictions directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--pH", type=str, default='5.0',
                        help="pH value to use for ground truth selection")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Path to the ground truth MD directory")
    parser.add_argument("--filter_proteins", type=str, nargs='+', default=None,
                        help='Optional list of protein names to filter the dataset')
    parser.add_argument("--max_samples_per_protein", type=int, default=None,
                        help='Maximum number of samples to load per protein')
    parser.add_argument("--max_sequence_length", type=int, default=None,
                        help='Maximum sequence length to include in the dataset')

    args = parser.parse_args()

    # print input parameters before running
    print("Input parameters:")
    print(args)
    
    datasets = build_dataset(
        predictions_dir=args.predictions_dir,
        pH=args.pH,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        filter_proteins=args.filter_proteins,
        max_samples_per_protein=args.max_samples_per_protein,
        max_sequence_length=args.max_sequence_length,
        split_dataset=True
    )

if __name__ == "__main__":
    main()
    