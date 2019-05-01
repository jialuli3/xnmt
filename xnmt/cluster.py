"""Implements the k-means algorithm.
"""

import numpy as np
import dynet as dy

from typing import List, Union, Optional
from pypinyin import pinyin,lazy_pinyin,Style
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
        self._curr_occupied_clusters=dict(zip(range(self._n_dims),[0]*self._n_dims))
        #self._centroid=self._mu.npvalue()
        self._characters_dist={}
        self.convert_dictionary("/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char","/home/jialu/xnmt/recipes/las-tedlium/vocab.char",False)
        self.char_cluster_count=dict()
        self.cluster_char_count=dict()
        self.cluster_py_count=dict()

        self.COUNT=0
        self.CONTEXT=1
        self.PRIOR=2
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

    def record_charclustercount(self,char,cluster):
        if not isinstance(char,int):
            #print(char)
            if char[0]==' ':
                char_str="".join([' ',self.char_list[char[1]],self.char_list[char[2]]])
            elif char[2]==' ':
                char_str="".join([self.char_list[char[0]],self.char_list[char[1]],' '])
            else:
                char_str="".join([self.char_list[char[0]],self.char_list[char[1]],self.char_list[char[2]]])
            #print(char_str)
            # if char_str not in self.char_cluster_count:
            #     self.char_cluster_count[char_str]=dict()
            # if cluster not in self.char_cluster_count[char_str]:
            #     self.char_cluster_count[char_str][cluster]=0
            # self.char_cluster_count[char_str][cluster]+=1

            if cluster not in self.cluster_char_count:
                self.cluster_char_count[cluster]=dict()
            if char_str not in self.cluster_char_count[cluster]:
                self.cluster_char_count[cluster][char_str]=0
            self.cluster_char_count[cluster][char_str]+=1
        else:
            char=char-55
            v=self.vocab_list[char]
            py_str=self.vocab_dict[v][0]

            # if py_str not in self.char_cluster_count:
            #     self.char_cluster_count[char_str]=dict()
            # if cluster not in self.char_cluster_count[char_str]:
            #     self.char_cluster_count[char_str][cluster]=0
            # self.char_cluster_count[char_str][cluster]+=1

            if cluster not in self.cluster_py_count:
                self.cluster_py_count[cluster]=dict()
            if py_str not in self.cluster_py_count[cluster]:
                self.cluster_py_count[cluster][py_str]=0
            self.cluster_py_count[cluster][py_str]+=1

    def calc_impurity(self,task_id):
        """
        calc impurity loss as one training criteria per batch.
        cluster impurity: sum k (1-q(k))=sum k (1-p(k|i)p(i)/(sum j p(k|j)p(j)))       
        """
        char_loss=0
        total_char_count=0
        cluster_loss=0
        total_count=[]

        if task_id==0:
            cluster_char_count=self.cluster_char_count
        else:
            cluster_char_count=self.cluster_py_count

        for cluster in cluster_char_count:
            curr_total_count=sum(cluster_char_count[cluster].values())
            total_count.append(curr_total_count)
            if len(cluster_char_count[cluster].keys())>5:
                curr_loss=(curr_total_count-sum(sorted(cluster_char_count[cluster].values(),reverse=True)[0:5]))/curr_total_count
                char_loss+=curr_loss
                total_char_count+=1

        if total_char_count!=0:
            char_loss/=total_char_count
        total_count=sorted(total_count,reverse=True)
        cluster_loss=sum(total_count[0:50])/sum(total_count)

        #print("char_loss: "+str(char_loss)+" py_loss: "+str(py_loss))
        #return char_loss, py_loss, (cluster_char_loss+cluster_py_loss)/2
        return char_loss, cluster_loss

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
        # for i in range(len(correct_characters_index)):
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
        #return loss_out2cluster_diff_char

    def calc_loss_one_step_in2cluster(self):
        """
        Calculate entropy loss given input2cluster attention vectors
        """
        loss_input2cluster=0
        for i in range(len(self.att_in2cluster_vecs-1)):
            curr_loss=dy.sum_batches(dy.sum_elems(dy.cmult(self.att_in2cluster_vecs[i],dy.log(self.att_in2cluster_vecs[i+1]))))
            loss_input2cluster+=curr_loss.value()
        return -loss_input2cluster

    def query_cluster(self):
        sorted_cluster=sorted(self._curr_occupied_clusters.items(), key=lambda kv:kv[1])
        occupied_clusters=[k[0] for k in sorted_cluster if k[1]!=0] #occupied cluster index
        #unoccupied_clusters=[c for c in list(range(self._n_dims)) if c not in occupied_clusters] #unoccupied cluster index
        for cl in occupied_clusters:
            print("current occupied cluster index is "+str(cl))
            if cl in self.cluster_char_count:
                sorted_cl=sorted(self.cluster_char_count[cl].items(),key=lambda kv:kv[1], reverse=True)
                print(str(sum(self.cluster_char_count[cl].values()))+" "+str(sorted_cl[0:5]))
            if cl in self.cluster_py_count:
                sorted_cl=sorted(self.cluster_py_count[cl].items(),key=lambda kv:kv[1], reverse=True)
                print(str(sum(self.cluster_py_count[cl].values()))+" "+str(sorted_cl[0:5]))
            print("\n")

        char_loss,cluster_loss=self.calc_impurity(0)
        print("char_impurity: "+str(char_loss)+" cluster_percentage: "+str(cluster_loss))
        if len(self.cluster_py_count)!=0:
            py_loss,cluster_loss=self.calc_impurity(1)
            print("py_impurity: "+str(py_loss)+" cluster_percentage: "+str(cluster_loss))
        curr_occupied_clusters_num=len(occupied_clusters)
        print("There are currently "+str(curr_occupied_clusters_num)+" clusters occupied.")

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
            curr_feature=self._centroid[:,occupied_clusters[curr_cluster_index]]
            #Perform bisecting splitting
            self._centroid[:,occupied_clusters[curr_cluster_index]]=0.99*curr_feature
            self._centroid[:,unoccupied_clusters[curr_cluster_index]]=1.01*curr_feature
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
