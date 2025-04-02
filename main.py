import fire

from stocks_prediction.train_loop import train


def main():
    fire.Fire(
        {
            "train_all": train,
        }
    )


if __name__ == "__main__":
    main()
