import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import sklearn
from sklearn import metrics
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from utils import utils
import logging

logger = logging.getLogger(__name__)
logger = utils.get_logger(logger=logger)


class text_cluster_model():
    """class responsible for doing the clusterization model
      and all the steps before, excluding data preparation
    """    
    def __init__(self, dataset: list, plot_results: bool = False) -> None:
        """Constructor method for text_cluster_model class

        Parameters
        ----------
        dataset : list
            List with the texts that will be analysed
        plot_results : bool, optional
            Flag to say if the plots will show or not for the user.
            It needs to be specifically if using a notebook, by default False
        """        
        self.dataset = dataset
        self.results = dict()
        self.plot_results = plot_results

    @classmethod
    def __plot_lsa__(self, svd: TruncatedSVD) -> plt.figure:
        """Build a barplot to allow the visual 
        analysis when reducing dimensions


        Parameters
        ----------
        svd : TruncatedSVD
            Instance of the truncatedSVD model fitted

        Returns
        -------
        plt.figure
            The barplot with the number of features and the explained variance ratio
        """        
        features  = range(1, len(svd.components_))

        fig = plt.figure(figsize=(8, 6))
        plt.bar(features, svd.explained_variance_ratio_[1:])
        plt.xlabel('SVD Features')
        plt.ylabel('Variances')
        plt.title('Variance per SVD Feature')

        return fig

    @classmethod
    def __plot_opt_components_number__(self, x_tfidf: scipy.sparse.csr.csr_matrix, svd) -> plt.figure:
        """Build a lineplot to allow the visual 
        analysis to find the optimized number of components

        Parameters
        ----------
        x_tfidf : scipy.sparse.csr.csr_matrix
            Sparse matrix that represents the features extracted using Tf Idf.
        svd : TruncatedSVD
            Instance of the truncatedSVD model fitted

        Returns
        -------
        plt.figure
            The lineplot comparing the number of components and the cumulative variance
            in order to find the optmized number of components SVD.
        """                
        fig = plt.figure(figsize=(8, 6))

        # Alternative way to upgrade the procedure performed above
        # ToDo: Verificar o x_tfidf está correto, montar um if futuramente.
        plt.plot(range(0, x_tfidf.shape[1]-1), svd.explained_variance_ratio_.cumsum()*100, 
                        marker= 'o', linestyle = '--', color = 'b')

        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Sum of Variance (%)')
        plt.title('Optimized number of components SVD')

        plt.axhline(y= 80, color='r', linestyle='-')
        plt.text(30, 82, '80% cut-off threshold', color = 'red', fontsize=10)

        return fig
    
    @classmethod
    def __plot_elbow__(self, ks: range, inertias: list)-> plt.figure:
        """Build a lineplot to allow the visual 
        analysis when optimizing the number of clusters

        Parameters
        ----------
        ks : range
            Number of clusters
        inertias : list
            Dysfunction metric

        Returns
        -------
        plt.figure
            The lineplot comparing the inertias and the number of clusters
            in order to find the optmized number of clusters.
        """        
        fig = plt.figure(figsize=(8, 6))
        plt.plot(ks, inertias, '-o');
        plt.xlabel('Número de Clusters')
        plt.ylabel('Distorção/Inércia')
        plt.title('Método de Elbow')
        plt.xticks(ks)
        
        return fig

    @classmethod
    def __plot_silhouette__(self, n_cluster, ss_indexes) -> plt.figure:
        """Build a lineplot to allow the visual 
        analysis when optimizing the number of clusters

        Parameters
        ----------
        n_cluster : _type_
            Number of clusters
        ss_indexes : _type_
            Silhouette scores

        Returns
        -------
        plt.figure
            The lineplot comparing the silhouette scores and the number of clusters
            in order to find the optmized number of clusters.
        """        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(n_cluster, ss_indexes, '-')
        plt.xlabel('n_cluster')
        plt.ylabel('ss_indexes')
        plt.title('Local Maxima')
        
        return fig

    def extract_features(self) -> scipy.sparse.csr.csr_matrix:
        """Extract the text features using the tfidf vectorizer and put them into a sparse matrix.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse matrix with features extracted
        TfidfVectorizer
            Vectorizer used in the text transformation
        """        
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=5)
        t0 = time()
        x_tfidf = vectorizer.fit_transform(self.dataset)

        logger.info(f"vectorization done in {time() - t0:.3f} s")
        logger.info(f"n_samples: {x_tfidf.shape[0]}, n_features: {x_tfidf.shape[1]}")
        logger.info(f"We find that around {round(x_tfidf.nnz / np.prod(x_tfidf.shape)*100, 2):.2f}% of the entries of the X_tfidf matrix are non-zero")

        return x_tfidf, vectorizer

    def reduce_dimensions(self, x_tfidf: scipy.sparse.csr.csr_matrix) -> scipy.sparse.csr.csr_matrix:
        """Reduce the dimensions of the matrix using the TruncatedSVD 

        Parameters
        ----------
        x_tfidf : scipy.sparse.csr.csr_matrix
            Sparse matrix that represents the features extracted using Tf Idf.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse matrix with reduced features
        TruncatedSVD.explained_variance_ratio_
            List with the explained variance ratio
        """        
        # Instance of SVD
        svd = TruncatedSVD(x_tfidf.shape[1] - 1)

        # Making a pipeline for applying LSA (dimensionality reduction), and normalizing the data
        pipeline = make_pipeline(svd, Normalizer(copy=False))
        
        # Pipeline Fit
        pipeline.fit(x_tfidf)

        if self.plot_results:
            # LSA plot - Explained Variance Ratio
            self.__plot_lsa__(svd).show()
            
            # Plotting the optimize number of components SVD
            self.__plot_opt_components_number__(x_tfidf, svd)
            
        tsvd_var_ratios = svd.explained_variance_ratio_

        return x_tfidf, tsvd_var_ratios
    
    def optimize_components_number(self, x_tfidf: scipy.sparse.csr.csr_matrix, var_ratio: np.ndarray, goal_var: float)->np.ndarray:
        """Gets the optimum number of components

        Parameters
        ----------
        x_tfidf : scipy.sparse.csr.csr_matrix
            Sparse matrix with the extracted features from the text
        var_ratio : np.ndarray
            List with the explained variance ratio obtained in the reduce_dimensions function
        goal_var : float
            Goal to reach in the total variance

        Returns
        -------
        np.ndarray
            Normalized features with optimized components
        sklearn.pipeline.Pipeline
            Pipeline used in the normalization and optimization
        """        
        # Set initial number of features and variance explained so far
        total_variance, opt_n_components = 0.0, 0
        
        for explained_variance in var_ratio:
            # Add the explained variance to the total
            total_variance += explained_variance
            # Add one to the number of components
            opt_n_components += 1
            
            # If we reach our goal level of explained variance
            if total_variance >= goal_var:
                break
        
        logger.info(f'Optimized number of components: {opt_n_components}')

        # Use the number of components
        lsa = make_pipeline(TruncatedSVD(n_components= opt_n_components), Normalizer(copy=False))
        t0 = time()
        x_lsa = lsa.fit_transform(x_tfidf)
        explained_variance = lsa[0].explained_variance_ratio_.sum()

        logger.info(f"LSA done in {time() - t0:.3f} s")
        logger.info(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
        return x_lsa, lsa
    
    def optimize_clusters_number(self, x_lsa: np.ndarray)->int:
        """Gets the optimum number of clusters using the silhouette score and elbow method.

        Parameters
        ----------
        x_lsa : np.ndarray
            Normalized features with optimized components

        Returns
        -------
        int
            Optimum number of clusters
        """        
        # Calculate the perfomance measures
        # perform Elbow Method
        ks = range(2, 16)
        inertias = []
        n_clusters = []
        ss_indexes = []

        for k in ks:
            # Criação da instância do K-Means com k clusters: model
            model = KMeans(n_clusters = k, random_state= 12345)
            
            # Fit do modelo ao dataset
            model.fit(x_lsa)

            # Inertia and labels append
            inertias.append(model.inertia_)
            labels = model.labels_

            ss_index = metrics.silhouette_score(x_lsa, labels)

            n_clusters.append(k)
            ss_indexes.append(ss_index)
        

        df_ss = pd.DataFrame({'n_cluster': n_clusters, 'ss_indexes': ss_indexes})

        # Perform local maxima method
        max_ind = argrelextrema(df_ss['ss_indexes'].to_numpy(), np.greater)
        opt_n_clusters = df_ss['n_cluster'].iloc[max_ind].max()

        if np.isnan(opt_n_clusters):
            opt_n_clusters = df_ss.loc[df_ss['ss_indexes'] == df_ss['ss_indexes'].max()]['n_cluster'].to_list()[0]

        if self.plot_results:            
            # Plot elbow results
            self.__plot_elbow__(ks, inertias).show()

            # Plotting silhouette method
            self.__plot_silhouette__(df_ss['n_cluster'], df_ss['ss_indexes']).show()
            

        else:
            logging.warning('This method needs to set text_cluster_model.plot equals True to enable the visual analysis')

        return opt_n_clusters
            
    def cluster_texts(self, x_lsa: np.ndarray, n_clusters: int, lsa: sklearn.pipeline.Pipeline, vectorizer: TfidfVectorizer)->None:
        """Fit the k-means model to cluster the features used allong the class

        Parameters
        ----------
        x_lsa : np.ndarray
            Normalized features with optimized components
        n_clusters : int
            Number of clusters to be used in the model
        lsa : sklearn.pipeline.Pipeline
            Pipeline used to normalize the features
        vectorizer : TfidfVectorizer
            Vectorizer used in the features extraction
        """        

        # K-Means instance
        kmeans = KMeans(n_clusters= n_clusters, random_state= 12345)

        # Model Fit
        kmeans.fit(x_lsa)

        original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        for i in range(n_clusters):
            print(f"Cluster {i}: ", end="")
            for ind in order_centroids[i, :10]:
                print(f"{terms[ind]} ", end="")
            print("\n")

        return kmeans
    
    def format_results(self, raw_df: pd.DataFrame, x_lsa: np.ndarray, kmeans: sklearn.cluster._kmeans.KMeans):
        # Organizing the tools to store the models results
        results = list(kmeans.predict(x_lsa))
        d_results = {'index':[], 'predict':[]}
        index = 0

        # Sorting results
        for result in results:
            d_results["index"].append(index)
            d_results["predict"].append(result)
            index += 1

        # Creating keys 
        df_y = pd.DataFrame(d_results)
        raw_df["index"] = range(0,raw_df.shape[0])

        df_output = pd.merge(raw_df, df_y, on="index")
        df_output.drop("index", axis=1, inplace=True)

        return df_output