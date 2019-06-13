"""Implements the k-means algorithm.
"""

import numpy as np
import dynet as dy

from typing import List, Union, Optional
from pypinyin import pinyin,lazy_pinyin,Style
from math import ceil
from copy import deepcopy
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import pairwise_distances_argmin 
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xnmt.attender import Attender, MlpAttender
from xnmt.param_collection import ParamManager
from xnmt.param_init import NormalInitializer, ParamInitializer
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt.expression_sequence import ExpressionSequence


class Cluster(object):
    # def fit(self, x: np.ndarray):
    #     """
    #         Compute clustering for the specific method.
    #     """
    #     raise NotImplementedError('fit must be implemented by subclasses of Cluster')

    # def predict(self) -> np.asarray:
    #     """
    #         Predict output label for input by using clustered centroid.
    #     """
    #     raise NotImplementedError('predict must be implemented by subclasses of Cluster')

    # def calc_loss(self,x:np.ndarray)-> dy.Expression:
    #     """
    #     Calc loss for current clustering method
    #     """
    #     raise NotImplementedError("calc_loss must be implemented by subclasses of Cluster")

    def initialize_cluster_counting(self):
        self._total_occupied_clusters=dict(zip(range(self._n_dims),[0]*self._n_dims))
        self.cluster_char_count={}
        self.char_cluster_count={}

    def initialize_curr_cluster_counting(self):
        self._curr_occupied_clusters=dict(zip(range(self._n_dims),[0]*self._n_dims))
        #self.curr_cluster_char_count={}
        #self.curr_char_cluster_count={}        

    def initialize_threshold(self,task_id):
        if task_id==0:
            self.cluster_per_char=2
            self.char_per_cluster=10
        else:
            self.cluster_per_char=2
            self.char_per_cluster=5

    def convert_dictionary(self,vocab,char,heter):
        """
        style: 0-no tone, 1 - tone
        """
        self.vocab_dict={}
        self.vocab_list=["SS","ES"]
        self.char_list=["SS","ES"]
        f=open(vocab,"r",encoding="GBK")
        vocabs = f.readlines()
        for v in vocabs:
            v=v.strip()
            if pinyin(v,heteronym=heter) == []:
                self.vocab_dict[v]=' '
            else:
                self.vocab_dict[v]=pinyin(v,heteronym=heter)[0]
            self.vocab_list.append(v)
        f.close()
        f=open(char,"r")
        chars=f.readlines()
        for ch in chars:
            ch=ch.strip()
            self.char_list.append(ch)
        f.close()

    def initialize_centroid(self,x):
        if self.initialized_centroid==False:
            farthest=pairwise_distances(x)
            farthest_idx=np.unique(np.squeeze(np.dstack(np.unravel_index(np.argsort(farthest.ravel())[::-1],farthest.shape)),axis=0))
            centroid_idx=farthest_idx[0:self._n_dims]
            self._centroid[0:len(centroid_idx),:]=x[np.asarray(centroid_idx),:]
            self.initialized_centroid=True

    def modify_cluster_record(self,char,old_cluster,new_cluster):
        self.cluster_char_count[old_cluster][char]-=1
        if self.cluster_char_count[old_cluster][char]==0:
            del self.cluster_char_count[old_cluster][char]
        self.char_cluster_count[char][old_cluster]-=1
        if self.char_cluster_count[char][old_cluster]==0:
            del self.char_cluster_count[char][old_cluster]
        
        if new_cluster not in self.cluster_char_count:
            self.cluster_char_count[new_cluster]=dict()
        if char not in self.cluster_char_count[new_cluster]:
            self.cluster_char_count[new_cluster][char]=0
        self.cluster_char_count[new_cluster][char]+=1
        if char not in self.char_cluster_count:
            self.char_cluster_count[char]=dict()
        if new_cluster not in self.char_cluster_count[char]:
            self.char_cluster_count[char][new_cluster]=0
        self.char_cluster_count[char][new_cluster]+=1

    def get_char_str(self,char):
        """
        convert char idx to char str
        char(int or tuple): int for mandarin character index; tuple for english trigraph
        """
        if not isinstance(char,int):
            #print(char)
            if char[0]==' ':
                char_str="".join([' ',self.char_list[char[1]],self.char_list[char[2]]])
            elif char[2]==' ':
                char_str="".join([self.char_list[char[0]],self.char_list[char[1]],' '])
            else:
                char_str="".join([self.char_list[char[0]],self.char_list[char[1]],self.char_list[char[2]]])
        else:
            char=char-55
            v=self.vocab_list[char]
            #py_str=self.vocab_dict[v][0]
            #char_str=v
            char_str=self.vocab_dict[v][0]
        return char_str

    def record_cluster_assignment_helper(self,char,cluster):
        self._total_occupied_clusters[cluster]+=1

        if char not in self.char_cluster_count:
            self.char_cluster_count[char]=dict()
        if cluster not in self.char_cluster_count[char]:
            self.char_cluster_count[char][cluster]=0
        self.char_cluster_count[char][cluster]+=1

        if cluster not in self.cluster_char_count:
            self.cluster_char_count[cluster]=dict()
        if char not in self.cluster_char_count[cluster]:
            self.cluster_char_count[cluster][char]=0
        self.cluster_char_count[cluster][char]+=1

    def record_cluster_assignment(self,cluster_assignments,char_idx):
        for i in range(len(cluster_assignments)):
            self.record_cluster_assignment_helper(char_idx[i],cluster_assignments[i])

    def calc_impurity(self):
        """
        calc impurity loss as one training criteria per batch.
        cluster impurity: sum k (1-q(k))=sum k (1-p(k|i)p(i)/(sum j p(k|j)p(j)))       
        """
        total_char_count=0
        impurity=0
        total_count=[]

        cluster_char_count=self.cluster_char_count

        for cluster in cluster_char_count:
            curr_total_count=sum(cluster_char_count[cluster].values())
            total_count.append(curr_total_count)
            if len(cluster_char_count[cluster].keys())>5:
                impurity+=(curr_total_count-sum(sorted(cluster_char_count[cluster].values(),reverse=True)[0:5]))/curr_total_count
                total_char_count+=1

        if total_char_count!=0:
            impurity/=total_char_count
        total_count=sorted(total_count,reverse=True)
        major_cluster_weight=0
        if sum(total_count)!=0:
            major_cluster_weight=sum(total_count[0:50])/sum(total_count)
        print("impurity of cluster: "+str(impurity))
        return impurity, major_cluster_weight

    def query_cluster(self):
        sorted_cluster=sorted(self._total_occupied_clusters.items(), key=lambda kv:kv[1])
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0] #occupied cluster index
        #sort_cluster_idx=occupied_clusters.sort()
        #unoccupied_clusters=[c for c in list(range(self._n_dims)) if c not in occupied_clusters] #unoccupied cluster index
        for cl in occupied_clusters:
            sorted_cl=sorted(self.cluster_char_count[cl].items(),key=lambda kv:kv[1], reverse=True)
            print("cluster_idx "+str(cl)+" count "+str(sum(self.cluster_char_count[cl].values()))\
                +" "+str(sorted_cl[0:10]))
            print("\n")
        
        for char in self.char_cluster_count.keys():
            if len(self.char_cluster_count[char])>self.cluster_per_char:
                print(char+str(self.char_cluster_count[char]))
            
        impurity,major_cluster_weight=self.calc_impurity()
        print("impurity: "+str(impurity)+" top 50 cluster weights: "+str(major_cluster_weight))
        curr_occupied_clusters_num=len(occupied_clusters)
        print("There are currently "+str(curr_occupied_clusters_num)+" clusters occupied.")

        # for char_str in self.cluster_vectors:
        #     if len(self.cluster_vectors[char_str].shape)>1:
        #         print(char_str)
        #         print("length"+str(self.cluster_vectors[char_str].shape))
        #         #print("mean vector is "+str(np.mean(self.cluster_vectors[char_str],axis=0)))
        #         print("covariance vector is ")
        #         print(np.cov(self.cluster_vectors[char_str]))

    def split_cluster(self):
        """
            Perform cluster splitting if not all clusters are occupied
            after each epoch.
        """
        #Calculate clusters to be split
        print("Performing cluster splitting...")
        sorted_cluster=sorted(self._curr_occupied_clusters.items(), key=lambda kv:kv[1], reverse=True)
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0]
        unoccupied_clusters=[c for c in list(range(self._n_dims)) if c not in occupied_clusters]
        #Bisecting K-means
        curr_cluster_index=0
        while len(occupied_clusters)<self._n_dims:
            curr_feature=self._centroid[occupied_clusters[curr_cluster_index],:]
            #Perform bisecting splitting
            self._centroid[occupied_clusters[curr_cluster_index],:]=0.999*curr_feature
            self._centroid[unoccupied_clusters[curr_cluster_index],:]=1.001*curr_feature
            occupied_clusters.append(unoccupied_clusters[curr_cluster_index])
            curr_cluster_index+=1
        self._mu.set_value(self._centroid)

class KMeans(Cluster,Serializable):
    """
    KMeans model
    """
    yaml_tag="!KMeans"
    @serializable_init
    def __init__(self,
            n_dims,
            n_components=50,
            max_iter=100,
            param_init: ParamInitializer = bare(NormalInitializer)
            )-> None:
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter


        self.pc = ParamManager.my_params(self)
        self._mu=self.pc.add_parameters((n_dims,n_components),init=param_init.initializer((n_dims,n_components)))
        self._centroid=self._mu.npvalue()     
        self.convert_dictionary("/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char","/home/jialu/xnmt/recipes/las-tedlium/vocab.char",False)
        self.initialize_cluster_counting()
        self.char_indexes=None
        self.prev_labels=[]
        self.prev_centers=None
        self.kmeans=SkKMeans(n_clusters=self._n_dims,random_state=0,n_init=5,tol=0.01)

    def fit(self,x):
        print("data length "+str(x.shape[0]))
        self.predict_labels=self.kmeans.fit_predict(x)
        self._centroid=self.kmeans.cluster_centers_
        self._mu.set_value(self._centroid)
        self.reorder_cluster_idx()

    def predict(self):
        return self.predict_labels

    def reorder_cluster_idx(self):
        curr_labels=self.predict()
        if self.prev_labels==[]:
            self.record_cluster_assignment(curr_labels,self.char_indexes)           
            self.prev_labels=curr_labels
        else:
            pargmin=pairwise_distances_argmin(self._centroid,self.prev_centers)#may duplicate    
            new_labels=[pargmin[curr_labels[i]] for i in range(len(curr_labels))]
            print("new_labels"+str(new_labels))
            print("curr_labels"+str(curr_labels))
            self.record_cluster_assignment(new_labels,self.char_indexes)
            self.prev_labels=new_labels

        self.prev_centers=deepcopy(self._centroid)
        self.query_cluster()
        self.initialize_cluster_counting()

    def calc_loss(self, x):
        """
            Calculate L2 distance for given hidden units of acoustic inputs and corresponding
            cluster centroid.
            Args:
                x(numpy.ndarray): Array containing the feature of dimension (N,
                    ndims).
            Returns:
                loss(float): the sum loss per hidden node.

        """
        loss=self.kmeans.inertia_
        return dy.constant(1,loss)

class KMeansSelfImplement(Cluster, Serializable):
    yaml_tag='!KMeansSelfImplement'
    @serializable_init
    def __init__(self,
            n_dims,
            n_components=50,
            max_iter=100,
            tol=0.01,
            param_init: ParamInitializer = bare(NormalInitializer)
            )-> None:
        """Initialize a KMeans model
        Args:
            n_dims(int): The dimension of the feature.
            n_components(int): The number of cluster in the model.
            max_iter(int): The number of iteration to run EM.
            loss_type(int: 1- entropy loss; 2 - L2 distance loss; 3- L2 distance loss + entropy loss
            neg_log_softmax(numpy array): log softmax of characters at each step
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self.tol=tol

        # Randomly initialize model parameters using Gaussian distribution of 0
        # mean and unit variance.
        self.pc = ParamManager.my_params(self)
        self._mu=self.pc.add_parameters((n_dims,n_components),init=param_init.initializer((n_dims,n_components)))
        self._centroid=self._mu.npvalue()
        self.convert_dictionary("/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char","/home/jialu/xnmt/recipes/las-tedlium/vocab.char",False)
        self.initialize_cluster_counting()
        self.initialized_centroid=False
        self.round=0
        self.cache_x=np.asarray([])
        self.cache_char=[]
        self.cache_clusters=[]
        self.visualized_center=False
        
    def evaluate(self,x,char_indexes):
        if self.visualized_center==False:
            self.visualize_center()
        self.round+=1
        labels=self.predict(x)
        char_str_list=[]
        for i in range(len(char_indexes)):
            char_str_list.append(self.get_char_str(char_indexes[i]))
        for i in range(len(labels)):
            self._total_occupied_clusters[labels[i]]+=1
        self.cache_data(x,char_str_list,labels)
        self.record_cluster_assignment(labels,char_str_list)
        #if self.round % 1000==0:
        #    self.visualize_data(self.cache_x,self.cache_clusters,self.cache_char)
        #self.query_cluster()

    def visualize_center(self):
        self.visualized_center=True
        pc_center=PCA(n_components=2)
        df_center=pd.DataFrame(pc_center.fit_transform(self._centroid),columns=["PCA1","PCA2"])
        ax=sns.scatterplot(x="PCA1",y="PCA2",data=df_center,legend=False)
        fig=ax.get_figure()
        if self._n_dims==3000:
            fig.savefig("/home/jialu/xnmt/recipes/las-tedlium/visualize_english_center.png")
        else:
            fig.savefig("/home/jialu/xnmt/recipes/las-tedlium/visualize_mandarin_center.png")
       
    def visualize_cluster(self):
        pca_2d=PCA(n_components=2)
        pd2=pd.DataFrame(pca_2d.fit_transform(self.cache_x),columns=["PCA1","PCA2"])

        pd2=pd2.assign(char=self.cache_char,cluster=self.cache_clusters)

        unique_clusters,color_inds=np.unique(self.cache_clusters,return_inverse=True)
        color_palette=sns.hls_palette(n_colors=len(unique_clusters))
        plt.figure(figsize=(20,20))

        #print(self.cache_x.shape,len(unique_clusters))
        for index, row in pd2.iterrows():
            plt.annotate(row["char"],(row["PCA1"],row["PCA2"]),\
                horizontalalignment='center',verticalalignment='center',size=15,\
                color=color_palette[color_inds[index]])

        if "the" in self.cache_char:
            plt.savefig("/home/jialu/xnmt/recipes/las-tedlium/visualize_english_2d_sequence0.png")
        else:
            plt.savefig("/home/jialu/xnmt/recipes/las-tedlium/visualize_mandarin_2d_sequence0.png")

        plt.clf()

        pd_center=PCA(n_components=2)
        df_center=pd.DataFrame(pd_center.fit_transform(self._centroid[unique_clusters,:]),columns=["PCA1","PCA2"])
        
        top_char=[]
        size=[]
        for cl in unique_clusters:
            sorted_cluster=sorted(self.cluster_char_count[cl].items(), key=lambda kv:kv[1], reverse=True)
            top_char.append(sorted_cluster[0][0])
            size.append(self._total_occupied_clusters[cl])
        df_center=df_center.assign(top_char=top_char,size=size)
        ax=sns.scatterplot(x="PCA1",y="PCA2",hue="top_char",data=df_center,legend=False,size="size")
        for i in range(len(unique_clusters)):
            ax.text(df_center.PCA1[i],df_center.PCA2[i],df_center.top_char[i],color='black',weight='light')

        fig=ax.get_figure()
        if "the" in self.cache_char:
            fig.savefig("/home/jialu/xnmt/recipes/las-tedlium/visualize_english_centroid_sequence0.png")           
        else:
            fig.savefig("/home/jialu/xnmt/recipes/las-tedlium/visualize_madarin_centroid_sequence0.png")           

    def visualize_data(self,x,cluster_labels,char):
        pca_2d=PCA(n_components=2)
        pd2=pd.DataFrame(pca_2d.fit_transform(x),columns=["PCA1","PCA2"])
        pd2=pd2.assign(char=char,cluster=cluster_labels)
        unique_clusters,color_inds=np.unique(cluster_labels,return_inverse=True)
        color_palette=sns.hls_palette(n_colors=len(unique_clusters))
        print(len(color_palette))

        df_center=pd.DataFrame(pca_2d.fit_transform(self._centroid[unique_clusters,:]),\
            columns=["center1","center2"])
        top_char=[]
        for cl in unique_clusters:
            sorted_cluster=sorted(self.cluster_char_count[cl].items(), key=lambda kv:kv[1], reverse=True)
            top_char.append(sorted_cluster[0][0])
        df_center=df_center.assign(top_char=top_char)
        plt.figure(figsize=(20,20))

        #print("iterating data points")
        # for i,row in pd2.iterrows():
        #     plt.scatter(row["PCA1"],row["PCA2"],c=color_palette[color_inds[i]],marker='.')
            #plt.text(row["PCA1"],row["PCA2"],char[i],fontsize=5)
        print("iterating center")
        for i,row in df_center.iterrows():
           plt.scatter(row["center1"],row["center2"],c=color_palette[i],marker="+")
           plt.text(row["center1"],row["center2"],top_char[i],fontsize=10)

        if self._n_dims==3000:
            plt.savefig("/home/jialu/xnmt/recipes/las-tedlium/figs/eng_data_point_round"+str(self.round)+str(".png"))
        else:
            plt.savefig("/home/jialu/xnmt/recipes/las-tedlium/figs/man_data_point_round"+str(self.round)+str(".png"))
        self.query_cluster()   
    def fit(self, x,char_indexes):
        """Runs EM step for max_iter number of times.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
            ndims).
        """
        print("data length:"+str(x.shape[0]))
        prev_loss=0
        self.initialize_curr_cluster_counting()

        for _ in range(self._max_iter):
            r_ik=self._e_step(x); #Update cluster assignment
            self._m_step(x,r_ik); #Update cluster mean
            curr_loss=self.calc_loss(x).value()
            print("before reassign loss: "+str(curr_loss))
            if np.absolute(curr_loss-prev_loss)<self.tol:
            #if np.absolute(curr_loss-prev_loss)<0.1:
                break
            prev_loss=curr_loss
        
        self.round+=1
        char_str_list=[]
        for i in range(len(char_indexes)):
            char_str_list.append(self.get_char_str(char_indexes[i]))
        curr_predict_labels=self.predict(x)
        self.cache_data(x,char_str_list,curr_predict_labels)
        self.record_cluster_assignment(curr_predict_labels,char_str_list)           
        self._mu.set_value(self._centroid)

        if self.round % 1==0:
            print("getting current data visualization...")
            self.visualize_data(self.cache_x,self.cache_clusters,self.cache_char)

        if self.round % 5 ==0:
            print("perform reassigning...")
            self.reassign()
            self.query_cluster()
        
        if self.round % 10 == 0:
            print("perform splitting")
            self.split_cluster_after_reassign()
            self.query_cluster()
            after_loss=self.calc_reassign_loss(self.cache_x).value()
            print("after split loss "+str(after_loss))

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
        r_ik = self.get_posterior(x)
    
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
        self.inter_results=np.ndarray(shape=(x.shape[0],self._n_dims))
        for i in range(x.shape[0]):
            self.inter_results[i,:]=np.sum(np.square(x[i,:][None,:]-self._centroid),axis=1)
        r_ik=np.argmin(self.inter_results,axis=1)
        #print("r_ik,",r_ik.shape,r_ik)
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

        # unique, counts=np.unique(r_ik,return_counts=True)
        # r_ik_sum=dict(zip(unique,counts))
        # self._centroid[unique,:]=np.zeros((len(unique),self._n_components))
        # for i in range(x.shape[0]):
        #     curr_k=r_ik[i]
        #     self._centroid[curr_k]+=x[i]/r_ik_sum[curr_k]
        r_ik=self.predict(x)
        for i in range(x.shape[0]):
            curr_k=r_ik[i]
            self._curr_occupied_clusters[curr_k]+=1
            lr=1/self._curr_occupied_clusters[curr_k]
            self._centroid[curr_k]=(1-lr)*self._centroid[curr_k]+lr*x[i,:]
        self._mu.set_value(self._centroid)

    def calc_loss(self, x):
        """
            Calculate L2 distance for given hidden units of acoustic inputs and corresponding
            cluster centroid.
            Args:
                x(numpy.ndarray): Array containing the feature of dimension (N,
                    ndims).
            Returns:
                loss(float): the sum loss per hidden node.

        """
        # dyx=dy.inputTensor(x)
        # loss=dy.zeros((1,))
        # for i in range(x.shape[0]):
        #     loss+=dy.min_dim(dy.sum_dim(dy.square(dy.select_rows(dyx,[i])-self._mu),[1]))
        # loss/=x.shape[0]
        #loss= np.sum(np.min(self.inter_results,axis=1))/x.shape[0]
        loss=0
        r_ik=self.predict(x)
        for i in range(x.shape[0]):
            loss+=np.sum(np.square(x[i,:]-self._centroid[r_ik[i],:]))
        if x.shape[0]!=0:
            loss/=x.shape[0]
        #print("loss",loss)
        return dy.constant(1,loss)

    def calc_reassign_loss(self,x):
        loss=0
        clusters=self.cache_clusters
        for i in range(x.shape[0]):
            loss+=np.min(np.sum(np.square(x[i,:][None,:]-self._centroid),axis=1))
        loss/=x.shape[0]
        #print("loss",loss)
        return dy.constant(1,loss)  

    def predict(self,x):
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
        return self.get_posterior(x)
    
    def cache_data(self,x,char_indexes,predict_clusters):
        if x.shape==(0,):
            return

        if self.cache_x.shape==(0,):
            self.cache_x=x
        else:
            self.cache_x=np.vstack((self.cache_x,x))
        
        self.cache_char+=char_indexes
        self.cache_clusters+=list(predict_clusters)

    def reassign(self):
        """
        Keep the largest count of examples from a given char, keep those
        reassign chars that originally assigned to any other clusters to the largest count example
        """
        cache_x=self.cache_x
        cache_char=deepcopy(self.cache_char)
        cache_clusters=self.cache_clusters
        uni, inds, counts=np.unique(cache_char,return_inverse=True,return_counts=True)        
        sort_count_idx=np.argsort(counts)[::-1]

        for i in range(len(counts)):
            curr_char=uni[sort_count_idx[i]]
            curr_char_idx=np.argwhere(inds==sort_count_idx[i])
            sorted_cluster=sorted(self.char_cluster_count[curr_char].items(),key=lambda kv:kv[1],reverse=True)
            top_clusters=[k for k,v in sorted_cluster][0:self.cluster_per_char] #top 2 clusters given a char

            for j in range(len(curr_char_idx)):
                x_idx=curr_char_idx[j][0]
                if cache_clusters[x_idx] not in top_clusters:
                    reassignment_idx=np.argmin(np.sum(np.square(cache_x[x_idx,:][None,:]-self._centroid[top_clusters,:])))
                    new_cluster=top_clusters[reassignment_idx]
                    self.reassign_helper3(x_idx,cache_clusters[x_idx],new_cluster)
        self._mu.set_value(self._centroid)

    def reassign_helper3(self,idx,old_cluster,new_cluster):
        """
        perform two centroid updates 
        """
        char=self.cache_char[idx]
        x=self.cache_x[idx,:]
        self.cache_clusters[idx]=new_cluster

        lr_old=1/self._total_occupied_clusters[old_cluster]        
        self._centroid[old_cluster,:]=(1+lr_old)*self._centroid[old_cluster,:]-lr_old*x
        self._total_occupied_clusters[old_cluster]-=1

        self._total_occupied_clusters[new_cluster]+=1
        lr_new=1/self._total_occupied_clusters[new_cluster]
        self._centroid[new_cluster,:]=(1-lr_new)*self._centroid[new_cluster,:]+lr_new*x

        self.modify_cluster_record(char,old_cluster,new_cluster)

    def split_cluster_after_reassign(self):
        """
        Perform bisecting of kmeans algorithm 
        """
        sorted_cluster=sorted(self._total_occupied_clusters.items(), key=lambda kv:kv[1], reverse=True)
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0]
        unoccupied_clusters=list(set(range(self._n_dims))-set(occupied_clusters))

        if unoccupied_clusters==[]:
            return
        unocc_idx=0
        for oc in occupied_clusters:
            #if len(list(self.cluster_char_count[oc].keys()))>self.char_per_cluster:
            if sum(self.cluster_char_count[oc].values())>1:
                self.bisect_cluster(oc,unoccupied_clusters[unocc_idx])
            unocc_idx+=1
            if unocc_idx>=len(unoccupied_clusters):
                break
        self._mu.set_value(self._centroid)

    def bisect_cluster(self,old_cluster,new_cluster):
        """
            perform bisect kmeans clustering on current cluster idx； modify records of the clusters
            Args:
            old_cluster(int): current cluster index
            new_cluster(int): new cluster index
            Returns:
            cluster_centroids(numpy array): bisect cluster centroids
        """

        x_idxes=np.argwhere(np.asarray(self.cache_clusters)==old_cluster)
        datapoint=self.cache_x[x_idxes[0][0],:]
        for i in range(1,len(x_idxes)):
            datapoint=np.vstack((datapoint,self.cache_x[x_idxes[i][0],:]))
        kmeans=SkKMeans(n_clusters=2)
        bisect_labels=kmeans.fit_predict(datapoint)
        cluster_centers=kmeans.cluster_centers_
        self._centroid[old_cluster,:]=cluster_centers[0,:]
        self._centroid[new_cluster,:]=cluster_centers[1,:]
        for i in range(len(bisect_labels)):
            curr_char=self.cache_char[x_idxes[i][0]]
            if bisect_labels[i]==1:
                self._total_occupied_clusters[new_cluster]+=1
                self._total_occupied_clusters[old_cluster]-=1
                self.cache_clusters[x_idxes[i][0]]=new_cluster
                self.modify_cluster_record(curr_char,old_cluster,new_cluster)
            
    def violate_constrain(self,cluster,char):
        """Given an assignment check if it violates the constrain
        constrain:
        -2 clusters per char
        -5 chars per cluster
        Args:
            cluster(int): cluster index for current assignment
            char(int or tuple): current char for assignment
        Returns:
            violate(bool): True- violation False- not violation
        """
        #top 2 clusters given a char
        sorted_count=sorted(self.char_cluster_count[char].items(),key=lambda kv:kv[1],reverse=True)
        top_clusters=[k for k,v in sorted_count][0:self.cluster_per_char] #top five clusters given a char
        total_num_cluster=len(list(self.char_cluster_count[char].keys()))
        if total_num_cluster>self.cluster_per_char and cluster not in top_clusters:
            return True

        #5 chars per cluster
        sorted_count=sorted(self.cluster_char_count[cluster].items(),key=lambda kv:kv[1],reverse=True)
        top_chars=[k for k,v in sorted_count][0:self.char_per_cluster] #top five chars given a cluster
        total_num_char=len(list(self.cluster_char_count[cluster].keys()))

        if total_num_char>self.char_per_cluster and char not in top_chars:
            return True
        return False

    def reassign_helper1(self,curr_x,x_idx,char):
        """
        Reassign x to satisfy the constrains and to the closest cluster
        Record the new assignment
        Args:
            curr_x(numpy.ndarray): current acoustic vector for reassignment
            x_idx(int): index of x for given data points
            char(int or tuple): current char for reassignment
        """
        sorted_count=sorted(self.char_cluster_count[char].items(),key=lambda kv:kv[1],reverse=True)
        all_clusters=[k for k,v in sorted_count]
        #reassign to those with top clustered centroids
        assignment=self.r_ik[x_idx]
        candidate_clusters=[]
        for i in range(len(all_clusters)):
            if all_clusters[i]==assignment:
                continue
            sorted_cluster_char=sorted(self.cluster_char_count[all_clusters[i]].items(),key=lambda kv:kv[1],reverse=True)
            top_chars=[k for k,v in sorted_cluster_char][0:self.char_per_cluster]
            if char in top_chars:
                candidate_clusters.append(all_clusters[i])

        if candidate_clusters!=[]:
            reassignment=np.argsort(np.sum(np.square(curr_x[None,:]-self._centroid[candidate_clusters,:]),axis=1))
            self.r_ik[x_idx]=candidate_clusters[reassignment[0]]
            self.modify_cluster_record(char,assignment,self.r_ik[x_idx])                
        else:
            occupied_cluster=set([k[0] for k in self._curr_occupied_clusters.items() if k[1]!=0])
            empty_cluster=list(set(list(range(self._n_dims)))-occupied_cluster)            
            if empty_cluster!=[]:
                reassignment=empty_cluster[0]
                self.r_ik[x_idx]=reassignment
                self.modify_cluster_record(char,assignment,reassignment)

    def reassign_helper2(self,curr_x,x_idx,char):
        """
        Reassign extra x to top clusters
        """
        assignment=self.r_ik[x_idx]
        sorted_count=sorted(self.char_cluster_count[char].items(),key=lambda kv:kv[1],reverse=True)
        top_clusters=[k[0] for k in sorted_count if k[1]!=assignment]
        reassignment=np.argsort(np.sum(np.square(curr_x[None,:]-self._centroid[top_clusters,:]),axis=1))
        self.r_ik[x_idx]=top_clusters[reassignment[0]]
        self.modify_cluster_record(char,assignment,self.r_ik[x_idx])
        # if self.violate_constrain(self.r_ik[x_idx],char):
        #     print(char)
        #     print(self.cluster_char_count[self.r_ik[x_idx]])
        #     print(self.char_cluster_count[char])
        # if len(self.char_cluster_count[char])>self.cluster_per_char:
        #     print(char)
        #     print(self.cluster_char_count[self.r_ik[x_idx]])
        #     print(self.char_cluster_count[char])


class GaussianMixtureModel(Cluster,Serializable):
    """
    GMM model
    """
    yaml_tag="!GaussianMixtureModel"
    @serializable_init
    def __init__(self,
            n_dims,
            n_components=50,
            max_iter=100,
            param_init: ParamInitializer = bare(NormalInitializer)
            )-> None:
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter


        self.pc = ParamManager.my_params(self)
        self._mu=self.pc.add_parameters((n_dims,n_components),init=param_init.initializer((n_dims,n_components)))
        self._centroid=self._mu.npvalue()     
        self.convert_dictionary("/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char","/home/jialu/xnmt/recipes/las-tedlium/vocab.char",False)
        self.initialize_cluster_counting()
        self.char_indexes=None
        self.gmm=GMM(n_components=self._n_dims,tol=0.01,max_iter=10, warm_start=True)

    def fit(self,x):
        self.gmm.fit(x)
        self.predict_labels=self.gmm.predict(x)
        self._centroid=self.gmm.means_
        self._mu.set_value(self._centroid)
        self.record_cluster_assignment(self.char_indexes)           

    def predict(self):
        return self.predict_labels

    def calc_loss(self, x):
        """
            Calculate L2 distance for given hidden units of acoustic inputs and corresponding
            cluster centroid.
            Args:
                x(numpy.ndarray): Array containing the feature of dimension (N,
                    ndims).
            Returns:
                loss(float): the sum loss per hidden node.

        """
        # dyx=dy.inputTensor(x)
        # loss=dy.zeros((1,))
        # for i in range(x.shape[0]):
        #     loss+=dy.min_dim(dy.sum_dim(dy.square(dy.select_rows(dyx,[i])-self._mu),[1]))
        # loss/=x.shape[0]
        #loss= np.sum(np.min(self.inter_results,axis=1))/x.shape[0]
        loss=0
        prob=self.gmm.predict_proba(x)
        max_prob=np.max(prob,axis=1)
        #print([i for i in max_prob if i<0.8])
        for i in range(x.shape[0]):
            loss+=np.sum(np.square(x[i,:][None,:]-self._centroid*prob[i,:][:,None]))
        loss/=x.shape[0]
        #print("loss",loss)
        return dy.constant(1,loss)

class FuzzyCMeans(Cluster, Serializable):
    yaml_tag='!FuzzyCMeans'
    @serializable_init
    def __init__(self,
            n_dims=400,
            n_components=512,
            max_iter=20,
            epislon=0.01,
            m=2,
            method=1,
            param_init: ParamInitializer = bare(NormalInitializer)
            )-> None:
        """Initialize a FuzzyKmeans model
        Args:
            n_dims(int): The dimension of the feature.
            n_components(int): The number of cluster in the model.
            max_iter(int): The number of iteration to run EM.
            method(int): 0-fuzzycmeans,1-kmeans
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self.epislon=epislon
        self.m=m
        # Randomly initialize model parameters using Gaussian distribution of 0
        # mean and unit variance.
        self.pc = ParamManager.my_params(self)
        self._mu=self.pc.add_parameters((n_dims,n_components),init=param_init.initializer((n_dims,n_components)))
        self._centroid=self._mu.npvalue()     
        self.convert_dictionary("/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char","/home/jialu/xnmt/recipes/las-tedlium/vocab.char",False)
        self.initialize_cluster_counting()
        self.initialized_centroid=False

    def initialize(self,x):
        """
        Initialize matrix W=[w_ij], where w_ij indicates the degree of data point i belongs to cluster j.
        """
        #self.U=np.random.random_sample((self._n_dims,x.shape[0]))
        #self.U=np.transpose(np.transpose(self.U)/np.sum(self.U,axis=0)[:,None])
        self.U=np.zeros((self._n_dims,x.shape[0]))
        self._u_step(x)        

    def fit(self, x):
        """Runs EM step for max_iter number of times.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
            n_component).
        """
        self.initialize(x)
        for _ in range(self._max_iter):
            self.prev_U=deepcopy(self.U)
            self._k_step(x) #Update cluster assignment
            self._u_step(x) #Update cluster mean
            #self.calc_loss(x)
            #print("max diff"+str(np.max(self.prev_U)-np.max(self.U)))
            if np.absolute(np.max(self.prev_U)-np.max(self.U))<self.epislon:
                break
        self._mu.set_value(self._centroid)

    def _k_step(self, x):
        """
        Update cluster centroid.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                n_component).
        """
        U_square=np.square(self.U) #n_dim,N
        numerator=np.matmul(U_square,x)
        self._centroid=numerator/np.sum(U_square,axis=1)[:,None]

    def _u_step(self, x):
        """
        Update probability vector.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                n_component).
        """
        length=x.shape[0]
        for i in range(length):
            #distances_norm=np.sqrt(np.sum(np.square(x[i,:][:,None]-self._centroid),axis=0)) #(n_dims)
            distances_norm=np.linalg.norm(x[i,:][None,:]-self._centroid,axis=1)+1e-6 #(n_dims)
            #print(distances_norm.shape)
            denom=np.zeros((self._n_dims))
            for k in range(self._n_dims):
                denom+=np.square(distances_norm/distances_norm[k])
            self.U[:,i]=1/denom
    
    def calc_loss(self, x):
        """
            Calculate L2 distance for given hidden units of acoustic inputs and corresponding
            cluster centroid.
            Args:
                x(numpy.ndarray): Array containing the feature of dimension (N,
                    ndims).
            Returns:
                loss(float): the sum loss per hidden node.

        """
        loss=0
        U_square=np.square(self.U)
        for i in range(x.shape[0]):
            distances_norm=np.linalg.norm(x[i,:][None,:]-self._centroid,axis=1)
            loss+=np.sum(U_square[:,i]*distances_norm)
        loss/=x.shape[0]
        return dy.constant(1,loss)

    def predict(self):
        """Predict a label for each example in x.
        Args:
            char(list of char): Array containing the char.
        Returns:
            w_ik(numpy.ndarray): Array containing the predicted label for each
                x, dimension (N,)
        """
        assignments=[]
        top_five_w_ik=np.transpose(np.sort(self.U,axis=0)[::-1][0:5,:])
        print("predict",top_five_w_ik)
        top_five_assignments=np.transpose(np.argsort(self.U,axis=0)[::-1][0:5,:])
        print("assignment",top_five_assignments)
        assignments = np.argmax(self.U,axis=0)
        return assignments



class ClusterAdapt(Cluster,Serializable):
    """
    Make cluster as second context vector
    """
    yaml_tag="!ClusterAdapt"
    @serializable_init
    def __init__(self,
            n_dims: int = 100,
            n_components: int = 50,
            threshold: float =0.01,
            attender_in2cluster: Attender=bare(MlpAttender),
            param_init: ParamInitializer = bare(NormalInitializer)
            )-> None:
        """Initialize a cluster parameters in graph
        Args:
            n_dims(int): The number of cluster in the model.
            n_components(int): The dimension of the feature.
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self.attender_in2cluster = attender_in2cluster
        self.pc = ParamManager.my_params(self)
        self._mu=self.pc.add_parameters((n_components,n_dims),init=param_init.initializer((n_components,n_dims)))
        self.attention_out2in_vecs=None
        self.input_hidden_nodes = None
        #self._centroid=self._mu.npvalue()
        self._characters_dist={}
        self.convert_dictionary("/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char","/home/jialu/xnmt/recipes/las-tedlium/vocab.char",False)
        self.char_cluster_count=dict()
        self.cluster_char_count=dict()
        self.cluster_py_count=dict()

        self.COUNT=0
        self.CONTEXT=1
        self.PRIOR=2

    def clear_stale_expressions(self):
        self._characters_dist.clear()

    def init_attender_cluster(self):
        """
        Initialization of cluster in attender
        """
        mu=dy.parameter(self._mu)
        self.attender_in2cluster.init_sent(mu)

    def calc_attention_in2cluster(self):
        """
        Calculate attention_input2cluster
        """
        input_length=self.input_hidden_nodes.dim()[0][1]
        batch_size=self.input_hidden_nodes.dim()[1]
        self.att_in2cluster_dy_expressions=[]
        #Calculate all attention vectors throughout the input vector
        for i in range(input_length):
            attention=self.attender_in2cluster.calc_attention(self.input_hidden_nodes[i])
            #print("att_in2cluster",np.argmax(attention.value(),axis=0))
            if self.att_in2cluster_dy_expressions==[]:
                self.att_in2cluster_dy_expressions=attention
            else:
                self.att_in2cluster_dy_expressions=dy.concatenate_cols([self.att_in2cluster_dy_expressions,attention])
        #print(self.att_in2cluster_dy_expressions.dim())

    def fit(self, x, output_time, batch_idx):
        """
        Compute context vector output2clutser, c(i,:), output vector at time i,
        c(i,k)=sum t att_out2in(i,t)*att_in2cluster(t,k)
        Args:
            x(dy.expressions): An output vector i, Array containing the feature of dimension (N,batchs_size)).
            correct_predicted_batch_index(np.ndarray): numpy array of correctly predicted batch index
        Returns:
            context_out2cluster(batch_size*# of clusters)
        """
        #input_length=self.input_hidden_nodes.dim()[0][1]
        att_out2in=self.attention_out2in_vecs[output_time] #att_out2in(i,t)=p(t|i)
        context_out2cluster=dy.sum_dim(dy.cmult(dy.transpose(dy.pick_batch_elems(att_out2in,[batch_idx])),dy.pick_batch_elems(self.att_in2cluster_dy_expressions,[batch_idx])),[1])
        return context_out2cluster

    def predict(self,context_out2cluster):
        """
        Predict cluster label given output vector
        c(i,k)=\sum t att_output2input(i,t)*attention_input2clutser(t,k)
        Args:
        x(dy.Expression): output vector i  (N,batchs_size)
        Return:
        cluster_label(int): cluster index given output vector i
        """
        return np.argmax(context_out2cluster.value()) #c(i,k)

    def calc_KL_loss_diff_cluster_helper(self,context,prior):
        """
        Calculate different char cluster loss helper
        """
        num_char=prior.dim()[0][0]
        context=dy.reshape(context,(1,self._n_dims))
        prior=dy.reshape(prior,(num_char,1))       
        diff_cluster_loss=dy.zeros((1,))
        #print(np.sort(context.value())[::-1][0:6])
        context_sort=np.argsort(context.value())[::-1][0][0:6]
        #print(context_sort,dy.select_cols(context,context_sort[1:6]).value())
        cluster=self.predict(context)
        max_char_dist=prior*dy.select_cols(context,[cluster])       
        char_dist=prior*dy.select_cols(context,context_sort[1:6])
        #print(max_char_dist.dim(),char_dist.dim())
        diff_cluster_loss=dy.sum_elems(dy.cmult(max_char_dist+1e-6,dy.log(char_dist+1e-6)))
        return diff_cluster_loss

    def calc_KL_loss_same_char_helper(self,context1,context2,prior1,prior2):
        """
        Calculate different char cluster loss helper
        """
        num_char=prior1.dim()[0][0]
        char_dist1=dy.reshape(prior1,(num_char,1))*dy.reshape(context1,(1,self._n_dims))
        char_dist2=dy.reshape(prior2,(num_char,1))*dy.reshape(context2,(1,self._n_dims))
        same_char_loss=dy.sum_elems(dy.cmult(char_dist1+1e-6,dy.log(char_dist2+1e-6)))
        return same_char_loss

    def calc_KL_loss(self):
        """
        Calculate KL divergence measurement
        max D(pk1||pk2)= min - D(pk1||pk2)=min sum i sum k2!=k1 
        Args:
        char_dist(dy.expression):prior distribution
        """
        same_cluster_loss=dy.zeros((1,))
        diff_cluster_loss=dy.zeros((1,))
        total_count=0
        for char in self._characters_dist.keys():
            context=self._characters_dist[char][self.CONTEXT]
            prior=self._characters_dist[char][self.PRIOR]
            count=self._characters_dist[char][self.COUNT]
            total_count+=count
            if count==1:
                diff_cluster_loss+=self.calc_KL_loss_diff_cluster_helper(context,prior)
            else:
                for i in range(count):
                    diff_cluster_loss+=self.calc_KL_loss_diff_cluster_helper(dy.select_cols(context,[i])\
                        ,dy.select_cols(prior,[i]))
                # for i in range(count-1):       
                #     for j in range(i+1,count):
                #         same_cluster_loss-=self.calc_KL_loss_same_char_helper(dy.select_cols(context,[i]),\
                #             dy.select_cols(context,[j]),dy.select_cols(prior,[i]),dy.select_cols(prior,[j]))
        if total_count!=0:
            diff_cluster_loss/=total_count
        return same_cluster_loss, diff_cluster_loss

    def calc_char_loss(self,correct_characters_index):
        """
        Calculate context vector loss
        """
        loss_out2cluster_same_char=dy.zeros((1,))
        loss_out2cluster_diff_char=dy.zeros((1,))        
        #total_same=0
        # for char in self._characters_dist.keys():
        #     if self._characters_dist[char][self.COUNT]>=2:
        #         context_out2cluster=self._characters_dist[char][self.CONTEXT]
        #         count=self._characters_dist[char][self.COUNT]
        #         for i in range(count-1):
        #             for j in range(i+1,count):
        #                 total_same+=1
        #                 curr_loss_out2cluster_same_char=dy.sum_elems(dy.cmult(dy.select_cols(context_out2cluster,[i])+1e-6\
        #                     ,dy.log(dy.select_cols(context_out2cluster,[j])+1e-6)))
        #                 loss_out2cluster_same_char-=curr_loss_out2cluster_same_char

        # if total_same!=0:
        #     loss_out2cluster_same_char/=total_same
        ### count all chars for distribution
        # total_diff=0
        # for i in range(len(correct_characters_index)-1):
        #     char1 = correct_characters_index[i]
        #     char1_context= self._characters_dist_context[char1]
        #     char1_count=self._characters_dist_index_count[char1]
        #     if char1_count==1:
        #         char1_context=dy.reshape(char1_context,(self._n_dims,1))
        #     for j in range(i+1,len(correct_characters_index)):
        #         char2 = correct_characters_index[j]
        #         char2_context= self._characters_dist_context[char2]
        #         char2_count= self._characters_dist_index_count[char2]
        #         if char2_count==1:
        #             char2_context=dy.reshape(char2_context,(self._n_dims,1))
        #         #print(char1_count,char2_count)
        #         for ch1 in range(char1_count):
        #             for ch2 in range(char2_count):
        #                 curr_loss_out2cluster_diff_char=-dy.sum_elems(dy.cmult(dy.select_cols(char1_context,[ch1]),\
        #                         dy.log(dy.select_cols(char2_context,[ch2]))))
        #                 #self.diff_char.append(curr_loss_out2cluster_diff_char.value())
        #                 #print(curr_loss_out2cluster_diff_char.value())
        #                 if curr_loss_out2cluster_diff_char.value()>0:
        #                     total_diff+=1
        #                     loss_out2cluster_diff_char-=curr_loss_out2cluster_diff_char
        # if total_diff!=0:
        #     loss_out2cluster_diff_char/=total_diff

        ### count chars with mean calculation for particular dimension
        for i in range(len(correct_characters_index)):
            char=correct_characters_index[i]
            char_count=self._characters_dist[char][self.COUNT]
            if char_count==1:
                context_vec=self._characters_dist[char][self.CONTEXT]
            else:
                context_vec=dy.mean_dim(self._characters_dist[char][self.CONTEXT],[1],False)
            if i==0:
                avg_context=context_vec
            else:
                avg_context=dy.concatenate_cols([avg_context,context_vec])

        total_diff=0
        for i in range(len(correct_characters_index)):
            for j in range(i+1,len(correct_characters_index)):
                curr_loss_out2cluster_diff_char=dy.sum_elems(dy.cmult(dy.select_cols(avg_context,[i])+1e-6,dy.log(dy.select_cols(avg_context,[j])+1e-6)))
                if curr_loss_out2cluster_diff_char.value()<0:
                    total_diff+=1
                    loss_out2cluster_diff_char+=curr_loss_out2cluster_diff_char
                    
        if total_diff!=0:
            loss_out2cluster_diff_char/=total_diff

        # total_diff=0
        # for i in range(len(correct_characters_index)-1):
        #     char1=correct_characters_index[i]
        #     char2=correct_characters_index[i+1]
        #     if self._characters_dist[char1][self.COUNT]<2:
        #         context1=self._characters_dist[char1][self.CONTEXT]
        #     else:
        #         context1=dy.select_cols(self._characters_dist[char1][self.CONTEXT],[0])
        #     if self._characters_dist[char2][self.COUNT]<2:
        #         context2=self._characters_dist[char2][self.CONTEXT]
        #     else:
        #         context2=dy.select_cols(self._characters_dist[char2][self.CONTEXT],[0])

        #     curr_loss_out2cluster_diff_char=dy.sum_elems(dy.cmult(context1+1e-6,dy.log(context2+1e-6)))
        #     if curr_loss_out2cluster_diff_char.value()<0:
        #         total_diff+=1
        #         loss_out2cluster_diff_char+=curr_loss_out2cluster_diff_char
                    
        # if total_diff!=0:
        #     loss_out2cluster_diff_char/=total_diff
        return loss_out2cluster_same_char,loss_out2cluster_diff_char

    def calc_loss_one_step_context(self, x, word_prob, output_time, correct_predicted_batch_index, correct_characters_index):
        """
        Calculate cross-entropy loss given the output vectors
        Args:
        x(dy.Expression): all rnn output states for decoder
        word_prob(dy.Expression): log probability of word distribution
        output_time(int): output time stamp
        correct_predicted_batch_index(nd.numpyarray): batch index of correct predicted characters
        correct_characters_index(list of int): list of index of correct predicted characters
        """

        if len(correct_predicted_batch_index) == 0:
            return dy.zeros((1,))
        for i in range(len(correct_predicted_batch_index)):
            curr_vector=dy.pick_batch_elems(x,[i])
            curr_context_out2cluster=self.fit(curr_vector, output_time, correct_predicted_batch_index[i])
            self._curr_occupied_clusters[self.predict(curr_context_out2cluster)]+=1
            curr_char=correct_characters_index[i]
            self.record_charclustercount(curr_char,self.predict(curr_context_out2cluster))
            curr_word_prob=dy.exp(dy.pick_batch_elems(word_prob,[correct_predicted_batch_index[i]]))
            if curr_char not in self._characters_dist.keys():
                self._characters_dist[curr_char]=[1,curr_context_out2cluster,curr_word_prob]
            else:
                self._characters_dist[curr_char][self.COUNT]+=1
                self._characters_dist[curr_char][self.CONTEXT]=dy.concatenate_cols([self._characters_dist[curr_char][self.CONTEXT],curr_context_out2cluster])
                self._characters_dist[curr_char][self.PRIOR]=dy.concatenate_cols([self._characters_dist[curr_char][self.PRIOR],curr_word_prob])

        loss_out2cluster_same_char,loss_out2cluster_diff_char=self.calc_char_loss(correct_characters_index)
        loss_kl_same_char,loss_kl_diff_char=self.calc_KL_loss()
        #return 0.2*loss_out2cluster_same_char+0.8*loss_out2cluster_diff_char+loss_kl_same_char+loss_kl_diff_char
        return loss_out2cluster_diff_char+loss_kl_diff_char

    def calc_loss_one_step_in2cluster(self):
        """
        Calculate entropy loss given input2cluster attention vectors
        """
        loss_input2cluster=0
        for i in range(len(self.att_in2cluster_vecs-1)):
            curr_loss=dy.sum_batches(dy.sum_elems(dy.cmult(self.att_in2cluster_vecs[i],dy.log(self.att_in2cluster_vecs[i+1]))))
            loss_input2cluster+=curr_loss.value()
        return -loss_input2cluster
