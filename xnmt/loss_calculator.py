import dynet as dy
import numpy as np
import math

from sklearn.mixture import GaussianMixture as GMM
from typing import Union

from xnmt.input import SimpleSentenceInput
from xnmt.loss import FactoredLossExpr
from xnmt.persistence import serializable_init, Serializable, Ref
from xnmt.vocab import Vocab
from xnmt.constants import INFINITY
from xnmt.transform import Linear
from xnmt.param_collection import ParamManager
import xnmt.evaluator
import xnmt.batcher

class LossCalculator(object):
  """
  A template class implementing the training strategy and corresponding loss calculation.
  """
  def calc_loss(self, translator, initial_state, src, trg):
    raise NotImplementedError()

  def remove_eos(self, sequence, eos_sym=Vocab.ES):
    try:
      idx = sequence.index(eos_sym)
      sequence = sequence[:idx]
    except ValueError:
      # NO EOS
      pass
    return sequence

class AutoRegressiveMLELoss(Serializable, LossCalculator):
  """
  Max likelihood loss calculator for autoregressive models.

  Args:
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """
  yaml_tag = '!AutoRegressiveMLELoss'
  @serializable_init
  def __init__(self, truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.truncate_dec_batches = truncate_dec_batches

  def calc_loss(self, translator: 'translator.AutoRegressiveTranslator',
                initial_state: 'translator.AutoRegressiveDecoderState',
                src: Union[xnmt.input.Input, 'batcher.Batch'],
                trg: Union[xnmt.input.Input, 'batcher.Batch']):
    dec_state = initial_state
    trg_mask = trg.mask if xnmt.batcher.is_batched(trg) else None
    losses = []
    seq_len = trg.sent_len()
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert single_trg.sent_len() == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    input_word = None
    #print("seq_len",seq_len)
    for i in range(seq_len):
      ref_word = AutoRegressiveMLELoss._select_ref_words(trg, i, truncate_masked=self.truncate_dec_batches)
      if self.truncate_dec_batches and xnmt.batcher.is_batched(ref_word):
        dec_state.rnn_state, ref_word = xnmt.batcher.truncate_batches(dec_state.rnn_state, ref_word)
      dec_state, word_loss = translator.calc_loss_one_step(dec_state, ref_word, input_word)
      if not self.truncate_dec_batches and xnmt.batcher.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      losses.append(0.8*word_loss)
      input_word = ref_word

    if self.truncate_dec_batches:
      loss_expr = dy.esum([dy.sum_batches(wl) for wl in losses])
    else:
      loss_expr = dy.esum(losses)
    #print("las_loss "+str(loss_expr.value()))
    return FactoredLossExpr({"mle": loss_expr})

  @staticmethod
  def _select_ref_words(sent, index, truncate_masked = False):
    if truncate_masked:
      mask = sent.mask if xnmt.batcher.is_batched(sent) else None
      if not xnmt.batcher.is_batched(sent):
        return sent[index]
      else:
        ret = []
        found_masked = False
        for (j, single_trg) in enumerate(sent):
          if mask is None or mask.np_arr[j, index] == 0 or np.sum(mask.np_arr[:, index]) == mask.np_arr.shape[0]:
            assert not found_masked, "sentences must be sorted by decreasing target length"
            ret.append(single_trg[index])
          else:
            found_masked = True
        return xnmt.batcher.mark_as_batch(ret)
    else:
      if not xnmt.batcher.is_batched(sent): return sent[index]
      else: return xnmt.batcher.mark_as_batch([single_trg[index] for single_trg in sent])

class CTCLoss(Serializable, LossCalculator):
  """
  calculate MLE loss of ctc decoder
  """
  yaml_tag= '!CTCLoss'

  @serializable_init
  def __init__(self, truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False)) -> None:
    self.truncate_dec_batches = truncate_dec_batches
    self.param_collection=ParamManager.my_params(self)

  def forward_recursion(self,t,s,softmax_mat,trg,blank):
    """
    maximum likelihood training loss
    """ 
    if s < 0:
        return -np.Inf

    # sub-problem already computed
    if not np.isnan(self.cache_forward[t][s]):
        return self.cache_forward[t][s]

    # initial values
    if t == 0:
        if s == 0:
            res = np.log(softmax_mat[blank][0].value())
            #print("initial t "+str(t)+" s "+str(s)+ " value "+str(res))
        elif s == 1:
            res = np.log(softmax_mat[trg[1]][0].value())
            #print("initial t "+str(t)+" s "+str(s)+ " value "+str(res))
        else:
            res = -np.Inf

        self.cache_forward[t][s] = res
        return res

    # recursion on s and t
    a = self.forward_recursion(t-1, s, softmax_mat, trg, blank)
    b = self.forward_recursion(t-1, s-1, softmax_mat, trg, blank)
    A = max(a,b)
    #print("A "+str(a)+" a "+str(a)+" b "+str(b))
    if np.isinf(A):
      res = - np.Inf
    else:
      sum_exp = math.exp(a-A)+math.exp(b-A)
      res = A + np.log(sum_exp) + np.log(softmax_mat[trg[s]][t].value())
    # in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done
    if trg[s] == blank or (s >= 2 and trg[s-2] == trg[s]):
        self.cache_forward[t][s] = res
        #print("blank or repeated label t "+str(t)+" s "+str(s)+ " value "+str(self.cache_forward[t][s]))
        return res

    # otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1
    c = self.forward_recursion(t-1,s-2,softmax_mat,trg,blank)
    A = max(a,b,c)
    if np.isinf(A):
      res = - np.Inf
    else:
      sum_exp=math.exp(a-A)+math.exp(b-A)+math.exp(c-A)
      res = A + np.log(sum_exp) + np.log(softmax_mat[trg[s]][t].value())
    self.cache_forward[t][s] = res
    #print("non blank non repeated label t "+str(t)+" s "+str(s)+ " value "+str(self.cache_forward[t][s]))
    return res

  def backward_recursion(self,t,s,softmax_mat,trg,blank):
    """
    maximum likelihood training loss
    """
    _,T = softmax_mat.dim()[0]

    if s > trg.sent_len()-1:
        return -np.Inf

    # sub-problem already computed
    if not np.isnan(self.cache_backward[t][s]):
        return self.cache_backward[t][s]

    # initial values
    if t == T-1:
        if s == trg.sent_len()-1:
            res = np.log(softmax_mat[blank][t].value())
            #print("initial t "+str(t)+" s "+str(s)+ " value "+str(res))
        elif s == trg.sent_len()-2:
            res = np.log(softmax_mat[trg[s]][t].value())
            #print("initial t "+str(t)+" s "+str(s)+ " value "+str(res))
        else:
            res = -np.Inf

        self.cache_backward[t][s] = res
        #print("t "+str(t)+" s "+str(s)+ " value "+str(self.cache_backward[t][s]))
        return res

    # recursion on s and t
    a=self.backward_recursion(t+1, s, softmax_mat, trg, blank)
    b=self.backward_recursion(t+1, s+1, softmax_mat, trg, blank)
    A= max(a,b)
    if np.isinf(A):
      res = - np.Inf
    else:
      sum_exp=math.exp(a-A)+math.exp(b-A)
      res = A + np.log(sum_exp) + np.log(softmax_mat[trg[s]][t].value())

    # in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done
    if trg[s] == blank or (s + 2 < trg.sent_len() and trg[s+2] == trg[s]):
        self.cache_backward[t][s] = res
        #print("t "+str(t)+" s "+str(s)+ " value "+str(self.cache_backward[t][s]))
        return res

    # otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1
    c = self.backward_recursion(t+1,s+2,softmax_mat,trg,blank)
    A = max(a,b,c)
    if np.isinf(A):
      res = - np.Inf
    else:
      sum_exp=math.exp(a-A)+math.exp(b-A)+math.exp(c-A)
      res = A + np.log(sum_exp) + np.log(softmax_mat[trg[s]][t].value())
    self.cache_backward[t][s] = res
    #print("t "+str(t)+" s "+str(s)+ " value "+str(self.cache_backward[t][s]))
    return res   

  def rescale_forward_backward_mat(self,mat):
    where_are_nans=np.isnan(mat)
    where_are_ninf=np.isinf(mat)
    mat[where_are_nans]=0
    mat[where_are_ninf]=0
    # #print("zero out mat",mat)
    # A=np.amax(np.where(mat<0,mat,-np.Inf),axis=1)
    # #print("A",A)
    # #exp_mat=np.where(mat<0,np.exp(mat),mat)
    # sum_exp_mat=np.exp(mat-A[:,None])
    # sum_exp_mat[where_are_nans]=0
    # sum_exp_mat[where_are_ninf]=0    
    # Z=A+np.log(np.sum(sum_exp_mat,axis=1))
    # #print("normalize",Z)
    # mat-=Z[:,None]
    # mat[where_are_nans]=0
    # mat[where_are_ninf]=0
    return mat

  def prep_trg_input(self,trg,blank):
    input_trg=[blank]
    for x in trg:
      if x!=1:
        input_trg.append(x)
        input_trg.append(blank)
    return SimpleSentenceInput(input_trg)
    
  def calc_loss_helper(self,softmax_mat:'dy.expression',
    trg: xnmt.input.Input,
    blank:int):
    """
    If inference mode, forward loss will be returned,
    else, cross-entropy loss will be returned
    """
    input_trg=self.prep_trg_input(trg,blank)
    _,T =softmax_mat.dim()[0]
    ctc_loss=dy.zeros((1,))
    mle_loss=dy.zeros((1,))
    #print("trg sent len "+str(input_trg.sent_len()))
    #print("T "+str(T))

    if input_trg.sent_len()>T*2+1:
      return ctc_loss,mle_loss

    self.cache_forward=np.full((T,input_trg.sent_len()),np.nan)
    self.cache_backward=np.full((T,input_trg.sent_len()),np.nan)

    #perform forward and backward algorithm
    #np.set_printoptions(precision=3,edgeitems=50)    

    forward_loss=self.forward_recursion(T-1,input_trg.sent_len()-1,softmax_mat,input_trg,blank)+\
      self.forward_recursion(T-1,input_trg.sent_len()-2,softmax_mat,input_trg,blank)

    backward_loss=self.backward_recursion(0,0,softmax_mat,input_trg,blank)+\
        self.backward_recursion(0,1,softmax_mat,input_trg,blank)

    #perform rescaling of forward and backward cache
    self.cache_forward_rescale=self.rescale_forward_backward_mat(self.cache_forward)
    #print("rescale forward",self.cache_forward)
    
    self.cache_backward_rescale=self.rescale_forward_backward_mat(self.cache_backward)
    #print("rescale backward",self.cache_backward)

    self.cache_sum=self.cache_forward_rescale+self.cache_backward_rescale

    # self.cache_para=self.param_collection.add_lookup_parameters(self.cache_forward.shape,\
    #  init=self.cache_sum)
    # for t in range(T):
    #   num_selected_s=np.squeeze(np.argwhere(self.cache_sum[t,:]<0))
    #   if num_selected_s.shape==():
    #     num_selected_s=num_selected_s.reshape((1,))
    #   den_selected_s=np.asarray(input_trg)[num_selected_s]
    #   unique_labels,indexes,counts=np.unique(den_selected_s,\
    #     return_inverse=True,return_counts=True)
    #   #print("unique"+str(unique_labels)+"indexes"+str(indexes)+"counts"+str(counts))
    #   for i in range(len(counts)):
    #     if counts[i]>1:
    #       #print("curr count"+str(counts[i]))
    #       all_repeated_indexes=np.argwhere(indexes==i)
    #       sum_forward_backward=np.sum(self.cache_sum[t][num_selected_s[all_repeated_indexes]])
    #       #print("sum_forward_backward"+str(sum_forward_backward))
    #       for s in num_selected_s[all_repeated_indexes]:
    #         self.cache_sum[t][s]=sum_forward_backward
    for t in range(T):
      num_selected_s=np.squeeze(np.argwhere(self.cache_sum[t,:]<0))
      if num_selected_s.shape==():
        num_selected_s=num_selected_s.reshape((1,))      
      den_selected_s=np.asarray(input_trg)[num_selected_s]
      for i in range(len(num_selected_s)):
        s=int(num_selected_s[i])
        #cross entropy loss
        #ctc_loss -= dy.exp(dy.constant((1,),self.cache_sum[t][s]))*dy.log(softmax_mat[int(den_selected_s[i])][t])

        #maximum likelihood loss
        ctc_loss += (dy.constant((1,),self.cache_sum[t][s])-dy.log(softmax_mat[int(den_selected_s[i])][t]))
        # print("gamma t"+str(self.cache_sum[t][s])+" softmax mat "\
        #   +str((softmax_mat[int(den_selected_s[i])][t]).value()))
        if t==0:
          mle_loss = -dy.constant((1,),self.cache_sum[t][s]-np.log(softmax_mat[int(den_selected_s[i])][t].value()))      
    return ctc_loss, mle_loss

  def calc_loss(self, translator: 'translator.AutoRegressiveTranslator',
                hidden_embeddings: 'dy.expression_sequence',
                trg: Union[xnmt.input.Input, 'batcher.Batch']):

    losses = []
    mle_losses =[]
    batch_size = hidden_embeddings.dim()[1]
    #print("trg len"+str(trg[0].sent_len())+" maxT "+str(hidden_embeddings.dim()[0][1]))

    for i in range(batch_size):
      probs=translator.scorer.calc_probs(dy.pick_batch_elem(hidden_embeddings,i))    
      blank_idx=translator.trg_reader.vocab_size()-1
      ctc_loss, mle_loss =self.calc_loss_helper(probs,trg[i],blank_idx)
      mle_losses.append(mle_loss)
      losses.append(ctc_loss)

    if self.truncate_dec_batches:
      loss_expr = dy.esum([dy.sum_batches(wl) for wl in losses])
    else:
      loss_expr = dy.esum(losses)
    mle_loss_expr = dy.esum(mle_losses)
    print("avg batch ctc loss "+str(loss_expr.value()/batch_size))
    print("avg batch mle loss "+str(mle_loss_expr.value()/batch_size))
    return FactoredLossExpr({"ctc": loss_expr, "mle": mle_loss_expr})

class AutoRegressiveClusterLoss(Serializable, LossCalculator):
  """
  Max likelihood loss calculator for autoregressive models.

  Args:
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """
  yaml_tag = '!AutoRegressiveClusterLoss'

  @serializable_init
  def __init__(self, truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False),\
     evaluate_mode: bool =False) -> None:
    self.truncate_dec_batches = truncate_dec_batches
    self.evaluate_mode=evaluate_mode

  def calc_loss(self, translator: 'translator.AutoRegressiveTranslator',
                initial_state: 'translator.AutoRegressiveDecoderState',
                src: Union[xnmt.input.Input, 'batcher.Batch'],
                trg: Union[xnmt.input.Input, 'batcher.Batch']):

    dec_state = initial_state
    trg_mask = trg.mask if xnmt.batcher.is_batched(trg) else None
    word_losses = []
    cluster_losses=[]
    batch_indexes=[]
    char_indexes=[]
    cluster_data_all=np.asarray([])
    seq_len = trg.sent_len()
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert single_trg.sent_len() == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    input_word = None
    for i in range(seq_len):
      ref_word = AutoRegressiveClusterLoss._select_ref_words(trg, i, truncate_masked=self.truncate_dec_batches)
      length=len(list(ref_word))
      if i!=0:
          ref_word_prev=AutoRegressiveClusterLoss._select_ref_words(trg, i-1, truncate_masked=self.truncate_dec_batches)
      else:
          ref_word_prev=[" "]*length
      if i!=seq_len-1:
        ref_word_later=AutoRegressiveClusterLoss._select_ref_words(trg, i+1, truncate_masked=self.truncate_dec_batches)
      else:
        ref_word_later=[" "]*length

      if self.truncate_dec_batches and xnmt.batcher.is_batched(ref_word):
        dec_state.rnn_state, ref_word = xnmt.batcher.truncate_batches(dec_state.rnn_state, ref_word)
      #dec_state, word_loss, cluster_loss = translator.calc_loss_one_step(dec_state, ref_word, ref_word_prev,ref_word_later,input_word, i)
      dec_state, word_loss, cluster_data, batch_idx, char_idx=translator.calc_loss_one_step(dec_state,src,True,\
           ref_word, ref_word_prev,ref_word_later,input_word, i, self.evaluate_mode)
      if len(batch_idx)!=0:
        batch_indexes+=(batch_idx)
        char_indexes+=(char_idx)        
        if cluster_data_all.shape==(0,):
          cluster_data_all=cluster_data
        else:
          cluster_data_all=np.vstack((cluster_data_all,cluster_data))
      if not self.truncate_dec_batches and xnmt.batcher.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      word_losses.append(word_loss)
      if translator.cluster_type==0:
        cluster_losses.append(cluster_loss)
      input_word = ref_word

    # if self.truncate_dec_batches:
    #   loss_expr_word = dy.esum([dy.sum_batches(wl) for wl in losses])
    # else:
    #   loss_expr_word = dy.esum(losses)
    loss_expr_cluster=dy.zeros((1,))

    #translator.cluster.initialize_centroid(cluster_data_all)
    translator.cluster.initialize_threshold(translator.task_id)
    if self.evaluate_mode:
      translator.cluster.evaluate(cluster_data_all,char_indexes)
    else:
      translator.cluster.fit(cluster_data_all,char_indexes)
    cluster_loss=translator.cluster.calc_loss(cluster_data_all)
    
    cluster_losses.append(cluster_loss)

    loss_expr_word=dy.esum(word_losses)
    loss_expr_cluster=dy.esum(cluster_losses)
    #translator.cluster.split_cluster()
    return FactoredLossExpr({"mle": loss_expr_word, "cluster":loss_expr_cluster})

  @staticmethod
  def _select_ref_words(sent, index, truncate_masked = False):
    if truncate_masked:
      mask = sent.mask if xnmt.batcher.is_batched(sent) else None
      if not xnmt.batcher.is_batched(sent):
        return sent[index]
      else:
        ret = []
        found_masked = False
        for (j, single_trg) in enumerate(sent):
          if mask is None or mask.np_arr[j, index] == 0 or np.sum(mask.np_arr[:, index]) == mask.np_arr.shape[0]:
            assert not found_masked, "sentences must be sorted by decreasing target length"
            ret.append(single_trg[index])
          else:
            found_masked = True
        return xnmt.batcher.mark_as_batch(ret)
    else:
      if not xnmt.batcher.is_batched(sent): return sent[index]
      else: return xnmt.batcher.mark_as_batch([single_trg[index] for single_trg in sent])

class AutoRegressiveKMeansLoss(Serializable, LossCalculator):
  """
  Self defined KMeansLoss loss calculator.

  Args:
    truncate_dec_batches: whether the decoder drops batch elements as soon as these are masked at some time step.
  """
  yaml_tag = '!AutoRegressiveKMeansLoss'
  @serializable_init
  def __init__(self, truncate_dec_batches: bool = Ref("exp_global.truncate_dec_batches", default=False),
        dev_evaluate: bool = False, test_evaluate: bool = False) -> None:
    self.truncate_dec_batches = truncate_dec_batches
    self.dev_evaluate = dev_evaluate
    self.test_evaluate = test_evaluate

  def perform_cluster_splitting(self,translator: 'translator.AutoRegressiveTranslator'):
      """
      Perform cluster splitting after each epoch.
      """
      translator.cluster.split_cluster()
      translator.cluster.initalize_cluster_counting()

  def calc_loss(self, translator: 'translator.AutoRegressiveTranslator',
                initial_state: 'translator.AutoRegressiveDecoderState',
                src: Union[xnmt.input.Input, 'batcher.Batch'],
                trg: Union[xnmt.input.Input, 'batcher.Batch']):
    #allHiddenNodes=open("/home/jialu/TIMIT/allHiddenNodes.csv","a")
    #allHiddenNodesPhoneLabel=open("/home/jialu/TIMIT/allHiddenNodesPhoneLabel.csv","a")
    #allMaximalAttendedNodes=open("/home/jialu/TIMIT/knn/allMaximalAttendedNodes.csv","a")
    #allMaximalAttendedNodesAttIdx=open("/home/jialu/TIMIT/knn/allMaximalAttendedNodesAttIdx.csv","a")

    dec_state = initial_state
    trg_mask = trg.mask if xnmt.batcher.is_batched(trg) else None
    losses = []
    atts=[]
    hidden_units=np.array([])

    seq_len = trg.sent_len()
    if xnmt.batcher.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert single_trg.sent_len() == seq_len # assert consistent length
        assert 1==len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j,i]==0) and single_trg[i]==Vocab.ES]) # assert exactly one unmasked ES token
    input_word = None
    for i in range(seq_len):
      ref_word = AutoRegressiveKMeansLoss._select_ref_words(trg, i, truncate_masked=self.truncate_dec_batches)
      if self.truncate_dec_batches and xnmt.batcher.is_batched(ref_word):
        dec_state.rnn_state, ref_word = xnmt.batcher.truncate_batches(dec_state.rnn_state, ref_word)
      if self.test_evaluate:
          dec_state, word_loss, hidden_units_per_frame, curr_att = translator.calc_loss_one_step_evaluate(dec_state, ref_word, input_word, None)
          atts.append(curr_att)
      else:
          dec_state, word_loss, hidden_units_per_frame = translator.calc_loss_one_step(dec_state, ref_word, input_word, None)
      #Gather hidden units to cluster for an entire batch
      #print(str(i)+" "+str(hidden_units_per_frame.size))
      if hidden_units.size == 0 and hidden_units_per_frame.size:
          hidden_units=hidden_units_per_frame
      elif hidden_units.size and hidden_units_per_frame.size:
          hidden_units=np.vstack((hidden_units,hidden_units_per_frame))
      if not self.truncate_dec_batches and xnmt.batcher.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      losses.append(word_loss)
      input_word = ref_word
    # Perform cluster for all frames in the batch
    print("There are "+str(hidden_units.shape)+" hidden units.")
    hidden_units=np.asarray(hidden_units)

    # if hidden_units.size:
    #     if not self.dev_evaluate and not self.test_evaluate:
    #         translator.cluster.fit(hidden_units)
        #else:
            #every hidden nodes vectors
            #r_ik_all_hidden=translator.cluster._e_step(np.transpose(translator.attender.curr_sent.as_tensor().npvalue()))
            #currHiddenNodes=np.asarray(np.transpose(translator.attender.curr_sent.as_tensor().npvalue()))
            #np.savetxt(allHiddenNodes,currHiddenNodes)
            #np.savetxt(allHiddenNodesPhoneLabel,np.asarray(r_ik_all_hidden),fmt='%2d')
            #print("all hidden units frames: "+str(list(r_ik_all_hidden)))
            #r_ik=translator.cluster._e_step(hidden_units)#maximally attended frames
            #np.savetxt(allMaximalAttendedNodes,hidden_units)
            #np.savetxt(allMaximalAttendedNodesAttIdx,np.asarray(atts),fmt='%2d')
            #print("attention frames: "+str(list(r_ik)))
            #print("attention indexes: "+str(atts))
        #cluster_loss=translator.cluster.calc_loss(hidden_units)
        #print("Current cluster loss is "+ str(cluster_loss))
        #allHiddenNodes.close()
        #allHiddenNodesPhoneLabel.close()
        #allMaximalAttendedNodes.close()
        #allMaximalAttendedNodesAttIdx.close()

    if self.truncate_dec_batches:
      loss_expr = dy.esum([dy.sum_batches(wl) for wl in losses])
    else:
      loss_expr = dy.esum(losses)
    #print(dy.constant(1,cluster_loss).npvalue())
    return FactoredLossExpr({"mle": loss_expr,"cluster":dy.constant(1,cluster_loss)})

  @staticmethod
  def _select_ref_words(sent, index, truncate_masked = False):
    if truncate_masked:
      mask = sent.mask if xnmt.batcher.is_batched(sent) else None
      if not xnmt.batcher.is_batched(sent):
        return sent[index]
      else:
        ret = []
        found_masked = False
        for (j, single_trg) in enumerate(sent):
          if mask is None or mask.np_arr[j, index] == 0 or np.sum(mask.np_arr[:, index]) == mask.np_arr.shape[0]:
            assert not found_masked, "sentences must be sorted by decreasing target length"
            ret.append(single_trg[index])
          else:
            found_masked = True
        return xnmt.batcher.mark_as_batch(ret)
    else:
      if not xnmt.batcher.is_batched(sent): return sent[index]
      else: return xnmt.batcher.mark_as_batch([single_trg[index] for single_trg in sent])

class ReinforceLoss(Serializable, LossCalculator):
  yaml_tag = '!ReinforceLoss'

  # TODO: document me
  @serializable_init
  def __init__(self, evaluation_metric=None, sample_length=50, use_baseline=False,
               inv_eval=True, decoder_hidden_dim=Ref("exp_global.default_layer_dim"), baseline=None):
    self.use_baseline = use_baseline
    self.inv_eval = inv_eval
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.FastBLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric

    if self.use_baseline:
      self.baseline = self.add_serializable_component("baseline", baseline,
                                                      lambda: Linear(input_dim=decoder_hidden_dim, output_dim=1))

  def calc_loss(self, translator, initial_state, src, trg):
    # TODO(philip30): currently only using the best hypothesis / first sample for reinforce loss
    # A small further implementation is needed if we want to do reinforce with multiple samples.
    search_output = translator.search_strategy.generate_output(translator, initial_state)[0]
    # Calculate evaluation scores
    self.eval_score = []
    for trg_i, sample_i in zip(trg, search_output.word_ids):
      # Removing EOS
      sample_i = self.remove_eos(sample_i.tolist())
      ref_i = self.remove_eos(trg_i.words)
      # Evaluating
      if len(sample_i) == 0:
        score = 0
      else:
        score = self.evaluation_metric.evaluate(ref_i, sample_i) * \
                (-1 if self.inv_eval else 1)
      self.eval_score.append(score)
    self.true_score = dy.inputTensor(self.eval_score, batched=True)
    # Composing losses
    loss = FactoredLossExpr()
    if self.use_baseline:
      baseline_loss = []
      losses = []
      for state, logsoft, mask in zip(search_output.state,
                                      search_output.logsoftmaxes,
                                      search_output.mask):
        bs_score = self.baseline(state)
        baseline_loss.append(dy.squared_distance(self.true_score, bs_score))
        loss_i = dy.cmult(logsoft, self.true_score - bs_score)
        losses.append(dy.cmult(loss_i, dy.inputTensor(mask, batched=True)))
      loss.add_loss("reinforce", dy.sum_elems(dy.esum(losses)))
      loss.add_loss("reinf_baseline", dy.sum_elems(dy.esum(baseline_loss)))
    else:
      loss.add_loss("reinforce", dy.sum_elems(dy.cmult(self.true_score, dy.esum(logsofts))))
    return loss

class MinRiskLoss(Serializable, LossCalculator):
  yaml_tag = '!MinRiskLoss'

  @serializable_init
  def __init__(self, evaluation_metric=None, alpha=0.005, inv_eval=True, unique_sample=True):
    # Samples
    self.alpha = alpha
    if evaluation_metric is None:
      self.evaluation_metric = xnmt.evaluator.FastBLEUEvaluator(ngram=4, smooth=1)
    else:
      self.evaluation_metric = evaluation_metric
    self.inv_eval = inv_eval
    self.unique_sample = unique_sample

  def calc_loss(self, translator, initial_state, src, trg):
    batch_size = trg.batch_size()
    uniques = [set() for _ in range(batch_size)]
    deltas = []
    probs = []

    search_outputs = translator.search_strategy.generate_output(translator, initial_state, forced_trg_ids=trg)
    for search_output in search_outputs:
      logprob = search_output.logsoftmaxes
      sample = search_output.word_ids
      attentions = search_output.attentions

      logprob = dy.esum(logprob) * self.alpha
      # Calculate the evaluation score
      eval_score = np.zeros(batch_size, dtype=float)
      mask = np.zeros(batch_size, dtype=float)
      for j in range(batch_size):
        ref_j = self.remove_eos(trg[j].words)
        hyp_j = self.remove_eos(sample[j].tolist())
        if self.unique_sample:
          hash_val = hash(tuple(hyp_j))
          if len(hyp_j) == 0 or hash_val in uniques[j]:
            mask[j] = -INFINITY
            continue
          else:
            # Count this sample in
            uniques[j].add(hash_val)
          # Calc evaluation score
        eval_score[j] = self.evaluation_metric.evaluate(ref_j, hyp_j) * \
                        (-1 if self.inv_eval else 1)
      # Appending the delta and logprob of this sample
      prob = logprob + dy.inputTensor(mask, batched=True)
      deltas.append(dy.inputTensor(eval_score, batched=True))
      probs.append(prob)
    sample_prob = dy.softmax(dy.concatenate(probs))
    deltas = dy.concatenate(deltas)
    risk = dy.sum_elems(dy.cmult(sample_prob, deltas))

    ### Debug
    #print(sample_prob.npvalue().transpose()[0])
    #print(deltas.npvalue().transpose()[0])
    #print("----------------------")
    ### End debug

    return FactoredLossExpr({"risk": risk})

