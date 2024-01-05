import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, threshold: float = 0.05):
        self.raw_data = data
        self.raw_labels = labels
        self.threshold = threshold
        self.low_variance_genes = self.find_low_variance_genes(threshold)
        self.filtered_data = self.filter_genes()
        self.pca_output_path = "output/Exploration_pca"
        self.perform_pca()

    def get_filtered_data(self):
        return self.filtered_data

    def find_low_variance_genes(self, threshold: float) -> list:
        labeled_data = pd.concat([self.raw_labels, self.raw_data], axis=1)
        transposed_data = labeled_data.set_index('Class').T
        gene_variances = transposed_data.var(axis=1)
        low_variance_genes = gene_variances[gene_variances < threshold].index.tolist()
        return low_variance_genes

    def filter_genes(self) -> pd.DataFrame:
        filtered_data = self.raw_data.drop(columns=self.low_variance_genes, errors='ignore')
        return filtered_data

    def perform_pca(self):
        label_encoder = LabelEncoder()
        class_labels = self.raw_labels['Class']
        encoded_labels = label_encoder.fit_transform(class_labels)

        class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.filtered_data)

        pca_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2'])
        pca_df['Class'] = encoded_labels

        pca_df['Class'] = pca_df['Class'].map(class_mapping)

        custom_palette = ['blue', 'red', 'green', 'purple', 'yellow']

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Component 1', y='Component 2', hue='Class', data=pca_df, palette=custom_palette)
        plt.title('PCA of Gene Expressions')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best')
        plt.savefig(self.pca_output_path)
