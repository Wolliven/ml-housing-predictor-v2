"""
"""
import sys
import argparse

def main() -> None:
    from ml_engine import predict
    parser = argparse.ArgumentParser(
        description="Use model to make a prediction",
        epilog="python predict.py [--model model.pkl] input.json/csv [--out output.json/csv]"
    )
    parser.add_argument("--model", type=str, required=False, help="Path to model to be used in the prediction")
    parser.add_argument("--out", type=str, required=False, help="Path to save the prediction results (JSON or CSV format)")
    parser.add_argument("input", type=str, help="Path to data to be used in the prediction")

    args = parser.parse_args()

    predict(args.input, model_path=args.model, output_path=args.out)
    print(f"Prediction completed. Results saved to {args.out if args.out else 'predictions.json'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)