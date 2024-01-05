import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    """
        A class to explore and visualize data using PCA.

        Attributes
        ----------
        raw_data : pd.DataFrame
            The raw data to be explored.
        raw_labels : pd.DataFrame
            The labels corresponding to the raw data.
        threshold : float
            The variance threshold for filtering genes.
        low_variance_genes : list
            The list of genes with variance below the threshold.
        filtered_data : pd.DataFrame
            The data after filtering low variance genes.
        pca_output_path : str
            The path to save the PCA plot.

        Methods
        -------
        get_filtered_data():
            Returns the filtered data.
        find_low_variance_genes(threshold: float) -> list:
            Finds and returns the genes with variance below the threshold.
        filter_genes() -> pd.DataFrame:
            Filters out the low variance genes from the raw data.
        perform_pca():
            Performs PCA on the filtered data and plots the result.
        """
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, threshold: float = 0.05):
        """
        Constructs all the necessary attributes for the DataExplorer object.

        Parameters
        ----------
        data : pd.DataFrame
        The raw data to be explored.
            labels : pd.DataFrame
        The labels corresponding to the raw data.
            threshold : float, optional
        The variance threshold for filtering genes (default is 0.05).
        """
        self.raw_data = data
        self.raw_labels = labels
        self.threshold = threshold
        self.low_variance_genes = self.find_low_variance_genes(threshold)
        self.filtered_data = self.filter_genes()
        self.pca_output_path = "output/Exploration_pca"
        self.perform_pca()

    def get_filtered_data(self):
        """
        Returns the filtered data.

        Returns:
        -------
        pd.DataFrame
            The filtered data.
        """
        return self.filtered_data

    def find_low_variance_genes(self, threshold: float) -> list:
        """
        Finds and returns the genes with variance below the threshold.

        Parameters
        ----------
        threshold : float
            The variance threshold.

        Returns
        -------
        list
            The list of genes with variance below the threshold.
        """
        labeled_data = pd.concat([self.raw_labels, self.raw_data], axis=1)
        transposed_data = labeled_data.set_index('Class').T
        gene_variances = transposed_data.var(axis=1)
        low_variance_genes = gene_variances[gene_variances < threshold].index.tolist()
        return low_variance_genes

    def filter_genes(self) -> pd.DataFrame:
        """
        Filters out the low variance genes from the raw data.

        :return:
        pd.DataFrame
            The data after filtering low variance genes.
        """
        filtered_data = self.raw_data.drop(columns=self.low_variance_genes, errors='ignore')
        return filtered_data

    def perform_pca(self):
        """
        Performs PCA on the filtered data and plots the result.
        """
        label_encoder = LabelEncoder()
        class_labels = self.raw_labels['Class']
        encoded_labels = label_encoder.fit_transform(class_labels)

        class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.filtered_data)

        pca_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2'])
        pca_df['Class'] = encoded_labels

        pca_df['Class'] = pca_df['Class'].map(class_mapping)

        custom_palette = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']


        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Component 1', y='Component 2', hue='Class', data=pca_df, palette=custom_palette)
        plt.title('PCA of Gene Expressions')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best')
        plt.savefig(self.pca_output_path)
