"""Implements the k-means algorithm.
"""

import numpy as np
import dynet as dy

from typing import List, Union, Optional
from math import ceil
from xnmt.attender import Attender, MlpAttender
from xnmt.param_collection import ParamManager
from xnmt.param_init import NormalInitializer, ParamInitializer
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt.expression_sequence import ExpressionSequence

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

class ClusterAdapt(Cluster,Serializable):
    """
    Make cluster as second context vector
    """
    yaml_tag="!ClusterAdapt"
    @serializable_init
    def __init__(self,
            n_dims: int = 100,
            n_components: int = 50,
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
        #self._mu=self.pc.add_parameters((n_components,n_dims),init=param_init.initializer((n_components,n_dims)))
        self._mu=self.pc.add_parameters((n_components,n_dims),init=param_init.initializer((n_dims,n_components)))
        self.attention_out2in_vecs=None
        self.input_hidden_nodes = None
        self.total_loss=0
        self._curr_occupied_clusters=dict(zip(range(self._n_dims),[0]*self._n_dims))
        #self._centroid=np.transpose(self._mu.npvalue())
        self._centroid=self._mu.npvalue()
        self._prev_centroid=self._centroid
        self._characters_dist_index_count={}
        self._characters_dist_context={}


    def clear_stale_expressions(self):
        self._characters_dist_index_count={}
        self._characters_dist_context={}

    def init_attender_cluster(self):
        """
        Initialization of cluster in attender
        """
        #self._mu.set_value(np.transpose(self._centroid))
        self._mu.set_value(self._centroid)
        self.attender_in2cluster.init_sent(dy.inputTensor(self._centroid))

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
        input_length=self.input_hidden_nodes.dim()[0][1]
        att_out2in=self.attention_out2in_vecs[output_time] #att_out2in(i,t)=p(t|i)
        #context_out2in2cluster=np.zeros((self._n_dims,input_length)) #c(k,t|i)=p(t,k|i)
        context_out2cluster=dy.sum_dim(dy.cmult(dy.transpose(dy.pick_batch_elems(att_out2in,[batch_idx])),dy.pick_batch_elems(self.att_in2cluster_dy_expressions,[batch_idx])),[1])

        #print(dy.sum_elems(context_out2cluster).value())
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

    def calc_loss_one_step_context(self, x, output_time, correct_predicted_batch_index, correct_characters_index):
        """
        Calculate cross-entropy loss given the output vectors
        Args:
        x(dy.Expression): all rnn output states for decoder
        output_time(int): output time stamp
        correct_predicted_batch_index(nd.numpyarray): batch index of correct predicted characters
        correct_characters_index(list of int): list of index of correct predicted characters
        """
        loss_out2cluster_same_char=dy.zeros((1,))
        loss_out2cluster_diff_char=dy.zeros((1,))
        if len(correct_predicted_batch_index) == 0:
            return dy.zeros((1,))
        #print("num_correct_predict_chars",len(correct_predicted_batch_index))

        for i in range(len(correct_predicted_batch_index)):
            curr_vector=dy.pick_batch_elems(x,[i])
            curr_context_out2cluster=self.fit(curr_vector, output_time, correct_predicted_batch_index[i])
            self._curr_occupied_clusters[self.predict(curr_context_out2cluster)]+=1
            curr_char=correct_characters_index[i]
            if curr_char not in self._characters_dist_index_count.keys() or self._characters_dist_index_count[curr_char]==0:
                self._characters_dist_index_count[curr_char]=1
                self._characters_dist_context[curr_char]=curr_context_out2cluster
            else:
                self._characters_dist_index_count[curr_char]+=1
                self._characters_dist_context[curr_char]=dy.concatenate_cols([self._characters_dist_context[curr_char],curr_context_out2cluster])


        for i in range(len(correct_characters_index)-1):
            char1=self._characters_dist_context[correct_characters_index[i]]
            char2=self._characters_dist_context[correct_characters_index[i+1]]
            char1_count=self._characters_dist_index_count[correct_characters_index[i]]
            if char1_count==1:
                loss_out2loss_out2cluster_diff_char+=dy.sum_elems(dy.cmult(char1,dy.log(char2)))
            else:
                for ch1 in range(char1_count):
                    print("ch1",dy.select_cols(char1,[ch1]).dim())
                    loss_out2cluster_diff_char+=dy.sum_elems(dy.cmult(dy.select_cols(char1,[ch1]),dy.log(char2)))

        for char,count in self._characters_dist_index_count.items():
            if count>=2:
                context_out2cluster=self._characters_dist_context[char]
                for i in range(count-1):
                    for j in range(i+1,count):
                        loss_out2cluster_same_char-=dy.sum_elems(dy.cmult(dy.select_cols(context_out2cluster,[i]),dy.log(dy.select_cols(context_out2cluster,[j]))))
                self._characters_dist_index_count[char]=0
                self._characters_dist_context[char]=[]
        #print("loss_out2cluster",loss_out2cluster.value())
        return loss_out2cluster_same_char+loss_out2cluster_diff_char

    def calc_loss_one_step_frames(self):
        """
        Calculate entropy loss given input2cluster attention vectors
        """
        loss_input2cluster=0
        for i in range(len(self.att_in2cluster_vecs-1)):
            curr_loss=dy.sum_batches(dy.sum_elems(dy.cmult(self.att_in2cluster_vecs[i],dy.log(self.att_in2cluster_vecs[i+1]))))
            loss_input2cluster+=curr_loss.value()
        return -loss_input2cluster

    def query_cluster(self):
        sorted_cluster=sorted(self._curr_occupied_clusters.items(), key=lambda kv:kv[1], reverse=True)
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0] #occupied cluster index
        unoccupied_clusters=[c for c in list(range(self._n_dims)) if c not in occupied_clusters] #unoccupied cluster index
        curr_occupied_clusters_num=len(occupied_clusters)
        print("There are currently "+str(curr_occupied_clusters_num)+" clusters occupied.")
        print([k for k in sorted_cluster if k[1]!=0])

    def split_cluster(self):
        """
            Perform cluster splitting if not all clusters are occupied
            after each epoch.
        """
        #Calculate clusters to be split
        print("performing split cluster")
        sorted_cluster=sorted(self._curr_occupied_clusters.items(), key=lambda kv:kv[1], reverse=True)
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0] #occupied cluster index
        unoccupied_clusters=[c for c in list(range(self._n_dims)) if c not in occupied_clusters] #unoccupied cluster index
        curr_occupied_clusters_num=len(occupied_clusters)
        print("There are currently "+str(curr_occupied_clusters_num)+" clusters occupied.")
        print([k for k in sorted_cluster if k[1]!=0])

        curr_cluster_index=0
        self._centroid=self._mu.npvalue()
        while len(occupied_clusters)<self._n_dims:
        #while curr_cluster_index<curr_occupied_clusters_num:
            curr_feature=self._centroid[occupied_clusters[curr_cluster_index],:]
            #Perform bisecting splitting
            self._centroid[occupied_clusters[curr_cluster_index],:]=0.999*curr_feature
            self._centroid[unoccupied_clusters[curr_cluster_index],:]=1.001*curr_feature
            occupied_clusters.append(unoccupied_clusters[curr_cluster_index])
            curr_cluster_index+=1
        #self._mu.set_value(np.transpose(self._centroid))
        self._mu.set_value(self._centroid)
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
            loss_type(int: 1- entropy loss; 2 - L2 distance loss; 3- L2 distance loss + entropy loss
            neg_log_softmax(numpy array): log softmax of characters at each step
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
        self.log_softmax=None
        self.initalize_cluster_counting()


    def assign_neg_logsoftmax(self,word_losses):
        """
        Helper function to assign negative softmax likelihood for each character per step.
        Args:
            word_losses(list of dynet expressions): negative softmax log likelihood for the character
            sequence with dim(sent_len*batch_len)
        """
        sen_len=len(word_losses)
        batch_len=len(word_losses[0].value())
        self.log_softmax=np.zeros((sen_len,batch_len))
        for i in range(sen_len):
            self.log_softmax[i]=word_losses[i].value()

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
        r_ik = self.get_posterior(x)
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
                loss(float): the sum loss per hidden node.

        """
        self._centroid=self._mu.npvalue()
        # inter_results=np.ndarray(shape=(x.shape[0],self._n_components))
        # for i in range(x.shape[0]):
        #     for j in range(self._n_components):
        #         inter_results[i][j]=np.sum(np.square(x[i]-self._centroid[j]))
        # #loss= np.sum(np.min(inter_results,axis=1))/x.shape[0]
        inter_results = self.get_posterior_helper()
        loss = np.sum(np.min(inter_results,axis=1))
        return loss

    def get_posterior_helper(self):
        """
        Helper function to calculate the l2 distance for all cluster labels for the K-means clustering.
        """
        inter_results=np.ndarray(shape=(x.shape[0],self._n_components))
        for i in range(x.shape[0]):
            for j in range(self._n_components):
                inter_results[i][j]=np.sum(np.square(x[i]-self._centroid[j]))
        return inter_results

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
        # r_ik = np.ndarray(shape=(x.shape[0],));
        # inter_results=np.ndarray(shape=(x.shape[0],self._n_components))
        # for i in range(x.shape[0]):
        #     for j in range(self._n_components):
        #         inter_results[i][j]=np.sum(np.square(x[i]-self._centroid[j]))
        inter_results=self.get_posterior_helper()
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
            self._centroid[unoccupied_clusters[curr_cluster_index],:]=1.01*curr_feature
            occupied_clusters.append(unoccupied_clusters[curr_cluster_index])
            curr_cluster_index+=1
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
