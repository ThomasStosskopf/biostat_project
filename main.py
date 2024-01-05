import pandas as pd

from lib.data_exploration import DataExplorer

class main:

    def __init__(self, data_filepath:str, labels_filepath:str, threshold:float=0.05):
        self.raw_data = pd.read_csv(data_filepath, index_col=0, header=0)
        self.raw_labels = pd.read_csv(labels_filepath, index_col=0, header=0)
        self.data_explorer = DataExplorer(self.raw_data, self.raw_labels, threshold)
        self.data_filtered = self.data_explorer.get_filtered_data()


    def main(self) -> None:
        print("Data filtered")


if __name__ == "__main__":
    data = "data/data.csv"
    labels = "data/labels.csv"
    final = main(data, labels)
    final.main()