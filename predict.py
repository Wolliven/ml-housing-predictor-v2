"""
"""
import sys
import argparse

def main() -> None:
    from ml_engine import predict
    parser = argparse.ArgumentParser(
        description="Use model to make a prediction",
        epilog="python predict.py --model model.pkl --input --input.json"
    )
    parser.add_argument("--model", type=str, required=False, help="Path to model to be used in the prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to data to be used in the prediction")

    args = parser.parse_args()

    result = predict(args.input, model_path=args.model)
    print(result["model"], result["input"], result["prediction"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)