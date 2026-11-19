import kagglehub


def main() -> None:
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print(f"Path to dataset files: {path}")


if __name__ == "__main__":
    main()
