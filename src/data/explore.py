import matplotlib.pyplot as plt
import pandas as pd

from ..config import LABELS
from .preprocess import load_preprocessed_data


def main() -> None:
    train_df, val_df, test_df = load_preprocessed_data()
    dfs_list = [train_df, val_df, test_df]
    dfs_names = ["Train", "Validation", "Test"]

    for name, df in zip(dfs_names, dfs_list):
        print(f"{name} samples: {len(df)}")
    print()

    count_missing_values(dfs_list, dfs_names)

    for label in LABELS:
        plot_value_counts(
            dfs_list, dfs_names, label, title=f"Raw Value Counts - {label}"
        )


def count_missing_values(dfs: list[pd.DataFrame], names: list[str]) -> None:
    for name, df in zip(names, dfs):
        print(f"{name} missing values:")
        print(df.isna().sum())
        print()


def plot_missing_values(
    dfs: list[pd.DataFrame],
    names: list[str],
    title: str = "Percentage of Missing Values",
) -> None:
    assert len(dfs) == len(names), "Each DataFrame must have a corresponding name."
    missing_values = {name: (df.isnull().mean() * 100) for name, df in zip(names, dfs)}
    missing_values_df = pd.DataFrame(missing_values)
    missing_values_df.plot(
        kind="bar",
        figsize=(12, 6),
        title=title,
    )
    plt.ylabel("Percentage Missing")
    plt.xlabel("Columns")
    plt.legend(title="DataFrame")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_value_counts(
    dfs: list[pd.DataFrame],
    names: list[str],
    column: str,
    normalize: bool = True,
    title: str = "",
) -> None:
    assert len(dfs) == len(names), "Each DataFrame must have a corresponding name."
    value_counts = [df[column].value_counts(normalize=normalize) for df in dfs]
    value_counts = {
        name: (df[column].value_counts(normalize=normalize))
        for name, df in zip(names, dfs)
    }
    value_counts_df = pd.DataFrame(value_counts)
    value_counts_df.plot(kind="bar", figsize=(12, 6), title=title)
    plt.show()


if __name__ == "__main__":
    main()
