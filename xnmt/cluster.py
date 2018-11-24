"""Implements the k-means algorithm.
"""

import numpy as np
import dynet as dy

from typing import List, Union, Optional
from math import ceil
from xnmt.param_collection import ParamManager
from xnmt.param_init import NormalInitializer, ParamInitializer
from xnmt.persistence import Serializable, serializable_init, bare, Ref


class Cluster(object):
    def fit(self, x: np.ndarray):
        """
            Compute clustering for the specific method.
        """
        raise NotImplementedError('fit must be implemented by subclasses of Cluster')

    def predict(self, x: np.ndarray) -> np.asarray:
        """
            Predict output label for input by using clustered centroid.
        """
        raise NotImplementedError('fit_predict must be implemented by subclasses of Cluster')


class KMeans(Cluster, Serializable):
    yaml_tag='!KMeans'
    @serializable_init
    def __init__(self,
            n_dims,
            n_components=50,
            max_iter=100,
            param_init: ParamInitializer = bare(NormalInitializer)
            )-> None:
        """Initialize a KMeans model
        Args:
            n_dims(int): The dimension of the feature.
            n_components(int): The number of cluster in the model.
            max_iter(int): The number of iteration to run EM.
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter

        # Randomly initialize model parameters using Gaussian distribution of 0
        # mean and unit variance.
        #self._mu = np.random.normal(size=(n_components,n_dims))  # np.array of size (n_components, n_dims)
        self.pc = ParamManager.my_params(self)
        self._mu=self.pc.add_parameters((n_components,n_dims),init=param_init.initializer((n_components,n_dims)))
        self._centroid=self._mu.npvalue()

        self.initalize_cluster_counting()

    def fit(self, x):
        """Runs EM step for max_iter number of times.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
            ndims).
        """
        for i in range(self._max_iter):
            r_ik=self._e_step(x); #Update cluster assignment
            self._m_step(x,r_ik); #Update cluster mean
        self._mu.set_value(self._centroid)
        unique, counts=np.unique(r_ik,return_counts=True)
        for i,j in zip(unique,counts):
            self._curr_occupied_clusters[i]+=j
        print("Currently there are " + str(unique) +" clusters.")

    def _e_step(self, x):
        """Update cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        r_ik = self.get_posterior(x);
        return r_ik

    def _m_step(self, x, r_ik):
        """Update cluster mean.

        Updates self_mu according to the cluster assignment.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """

        unique, counts=np.unique(r_ik,return_counts=True)
        r_ik_sum=dict(zip(unique,counts))
        #self._centroid=np.zeros((self._n_components,self._n_dims))
        self._centroid[unique,:]=np.zeros(self._n_dims)
        for i in range(x.shape[0]):
            curr_k=r_ik[i]
            self._centroid[curr_k]+=x[i]/r_ik_sum[curr_k]
        self._mu.set_value(self._centroid)

    def calc_loss(self, x):
        """
            Calculate L2 distance for given hidden units of acoustic inputs and corresponding
            cluster centroid.
            Args:
                x(numpy.ndarray): Array containing the feature of dimension (N,
                    ndims).
            Returns:
                loss(float): the average loss per hidden node.

        """
        self._centroid=self._mu.npvalue()
        inter_results=np.ndarray(shape=(x.shape[0],self._n_components))
        for i in range(x.shape[0]):
            for j in range(self._n_components):
                inter_results[i][j]=np.sum(np.square(x[i]-self._centroid[j]))
        #loss= np.sum(np.min(inter_results,axis=1))/x.shape[0]
        loss = np.sum(np.min(inter_results,axis=1))
        return loss

    def get_posterior(self, x):
        """Computes cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
            each example, dimension (N,).
        """
        self._centroid=self._mu.npvalue()
        r_ik = np.ndarray(shape=(x.shape[0],));
        inter_results=np.ndarray(shape=(x.shape[0],self._n_components))
        for i in range(x.shape[0]):
            for j in range(self._n_components):
                inter_results[i][j]=np.sum(np.square(x[i]-self._centroid[j]))
        r_ik=np.argmin(inter_results,axis=1)
        #print("r_ik,",r_ik.shape,r_ik)
        return r_ik

    def split_cluster(self):
        """
            Perform cluster splitting if not all clusters are occupied
            after each epoch.
        """
        #Calculate clusters to be split
        print("Performing cluster splitting...")
        sorted_cluster=sorted(self._curr_occupied_clusters.items(), key=lambda kv:kv[1], reverse=True)
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0]
        unoccupied_clusters=[c for c in list(range(self._n_components)) if c not in occupied_clusters]
        curr_occupied_clusters_num=len(occupied_clusters)
        print("There are currently "+str(curr_occupied_clusters_num)+" clusters are occupied.")
        #Bisecting K-means
        curr_cluster_index=0
        while len(occupied_clusters)<self._n_components:
            curr_feature=self._centroid[occupied_clusters[curr_cluster_index],:]
            #Perform bisecting splitting
            self._centroid[occupied_clusters[curr_cluster_index],:]=0.99*curr_feature
            self._centroid[unoccupied_clusters[curr_cluster_index],:]=0.01*curr_feature
            occupied_clusters.append(unoccupied_clusters[curr_cluster_index])
            curr_cluster_index+=1

        # curr_count=0 #count how many times clusters have been split
        # curr_cluster_index=0
        # print(clusters)
        # split_factor=[0.99,0.01]
        # if each_split_num > 2:
        #     random_list=list(np.random.rand(each_split_num-2))
        #     split_factor+=random_list
        # for i in range(self._n_components):
        #     if curr_cluster_index<len(clusters):
        #         curr_feature=self._centroid[clusters[curr_cluster_index],:]
        #     if self._curr_occupied_clusters[i] == 0:
        #         #print("Splitting current cluster "+str(clusters[curr_cluster_index]))
        #         self._centroid[i,:]=split_factor[curr_count]*curr_feature
        #         curr_count+=1
        #         if curr_count >= each_split_num:
        #             curr_cluster_index+=1
        #             curr_count=0
        #Split clusters itself
        # for i in range(curr_cluster_index):
        #     random_num=np.random.rand(1)
        #     self._centroid[clusters[i],:]*=random_num
        self._mu.set_value(self._centroid)


    def initalize_cluster_counting(self):
        """
            Initialize cluster counting to determine whether to split cluster later on.
        """
        self._curr_occupied_clusters=dict(zip(range(self._n_components),[0]*self._n_components))


    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.

        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.

        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.cluster_label_map = []
        label_count=np.zeros((self._n_components,self._n_components));#cluster assignment #, digit number
        r_ik=self.get_posterior(x);
        for i in range(x.shape[0]):
            label_count[r_ik[i]][int(y[i])]+=1
        for i in range(self._n_components):
            self.cluster_label_map.append(np.argmax(label_count[i]))
        #print(self.cluster_label_map)

    def predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
                x, dimension (N,)
        """

        r_ik = self.get_posterior(x)
        y_hat = []
        for i in range(x.shape[0]):
            y_hat.append(self.cluster_label_map[r_ik[i]])
        return np.array(y_hat)
