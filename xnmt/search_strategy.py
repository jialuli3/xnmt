from collections import namedtuple
import math
from typing import Callable, List, Optional, Sequence
import numbers

import dynet as dy
import numpy as np

import xnmt.batcher
from xnmt import logger
from xnmt.length_normalization import NoNormalization, LengthNormalization
from xnmt.persistence import Serializable, serializable_init, bare
from xnmt.vocab import Vocab


# Output of the search
# words_ids: list of generated word ids
# attentions: list of corresponding attention vector of word_ids
# score: a single value of log(p(E|F))
# logsoftmaxes: a corresponding softmax vector of the score. score = logsoftmax[word_id]
# state: a NON-BACKPROPAGATEABLE state that is used to produce the logsoftmax layer
#        state is usually used to generate 'baseline' in reinforce loss
# masks: whether the particular word id should be ignored or not (1 for not, 0 for yes)
SearchOutput = namedtuple('SearchOutput', ['word_ids', 'attentions', 'score', 'logsoftmaxes', 'state', 'mask'])
CTCSearchOutput=namedtuple('CTCSearchOutput', ['word_ids', 'score', 'logsoftmaxes', 'state', 'mask'])

class SearchStrategy(object):
  """
  A template class to generate translation from the output probability model. (Non-batched operation)
  """
  def generate_output(self, translator, dec_state,
                      src_length=None, forced_trg_ids=None):
    """
    Args:
      translator (Translator): a translator
      dec_state (AutoRegressiveDecoderState): initial decoder state
      src_length (int): length of src sequence, required for some types of length normalization
      forced_trg_ids (List[int]): list of word ids, if given will force to generate this is the target sequence
    Returns:
      List[SearchOutput]: List of (word_ids, attentions, score, logsoftmaxes)
    """
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')

class GreedySearch(Serializable, SearchStrategy):
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len (int): maximum number of tokens to generate.
  """

  yaml_tag = '!GreedySearch'

  @serializable_init
  def __init__(self, max_len=100):
    self.max_len = max_len

  def generate_output(self, translator, initial_state,
                      src_length=None, forced_trg_ids=None):
    # Output variables
    score = []
    word_ids = []
    attentions = []
    logsoftmaxes = []
    states = []
    masks = []
    # Search Variables
    done = None
    current_state = initial_state
    for length in range(self.max_len):
      prev_word = word_ids[length-1] if length > 0 else None
      current_output = translator.generate_one_step(prev_word, current_state)
      current_state = current_output.state
      if forced_trg_ids is None:
        word_id = np.argmax(current_output.logsoftmax.npvalue(), axis=0)
        if len(word_id.shape) == 2:
          word_id = word_id[0]
      else:
        if xnmt.batcher.is_batched(forced_trg_ids):
          word_id = [forced_trg_ids[i][length] for i in range(len(forced_trg_ids))]
        else:
          word_id = [forced_trg_ids[length]]
      logsoft = dy.pick_batch(current_output.logsoftmax, word_id)
      if done is not None:
        word_id = [word_id[i] if not done[i] else Vocab.ES for i in range(len(done))]
        # masking for logsoftmax
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        logsoft = dy.cmult(logsoft, dy.inputTensor(mask, batched=True))
        masks.append(mask)
      # Packing outputs
      score.append(logsoft.npvalue())
      word_ids.append(word_id)
      attentions.append(current_output.attention)
      logsoftmaxes.append(dy.pick_batch(current_output.logsoftmax, word_id))
      states.append(translator.get_nobp_state(current_state))
      # Check if we are done.
      done = [x == Vocab.ES for x in word_id]
      if all(done):
        break
    masks.insert(0, [1 for _ in range(len(done))])
    words = np.stack(word_ids, axis=1)
    score = np.sum(score, axis=0)
    return [SearchOutput(words, attentions, score, logsoftmaxes, states, masks)]

class BeamSearch(Serializable, SearchStrategy):
  """
  Performs beam search.
  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
    one_best: Whether to output the best hyp only or all completed hyps.
    scores_proc: apply an optional operation on all scores prior to choosing the top k.
                 E.g. use with :class:`xnmt.length_normalization.EosBooster`.
  """

  yaml_tag = '!BeamSearch'
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])

  @serializable_init
  def __init__(self,
               beam_size: numbers.Integral = 1,
               max_len: numbers.Integral = 100,
               len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True,
               scores_proc: Optional[Callable[[np.ndarray], None]] = None) -> None:
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc

  def generate_output(self,
                      translator: 'xnmt.models.translators.AutoRegressiveTranslator',
                      initial_state,
                      src_length: Optional[numbers.Integral] = None) -> List[SearchOutput]:
    active_hyp = [self.Hypothesis(0, None, None, None)]
    completed_hyp = []
    for length in range(self.max_len):
      if len(completed_hyp) >= self.beam_size:
        completed_hyp = sorted(completed_hyp, key=lambda hyp: hyp.score, reverse=True)
        completed_hyp = completed_hyp[:self.beam_size]
        worst_complete_hyp_score = completed_hyp[-1].score
        active_hyp = [hyp for hyp in active_hyp if hyp.score >= worst_complete_hyp_score]
        # Assumption: each additional word will always *decrease* the total score.
        if len(active_hyp) == 0:
          break

      # Expand hyp
      new_set = []
      for hyp in active_hyp:
        # Note: prev_word has *not* yet been added to prev_state
        if length > 0:
          prev_word = hyp.word
          prev_state = hyp.output.state
        else:
          prev_word = None
          prev_state = initial_state

        # We have a complete hyp ending with </s>
        if prev_word == Vocab.ES:
          completed_hyp.append(hyp)
          continue

        # Find the k best words at the next time step
        current_output = translator.add_input(prev_word, prev_state)
        top_words, top_scores = translator.best_k(current_output.state, self.beam_size, normalize_scores=True)

        # Queue next states
        for cur_word, score in zip(top_words, top_scores):
          assert len(score.shape) == 0
          new_score = self.len_norm.normalize_partial_topk(hyp.score, score, length + 1)
          new_set.append(self.Hypothesis(new_score, current_output, hyp, cur_word))

      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)

    # Take only the one best, if that's what was desired
    if self.one_best:
      hyp_and_score = [hyp_and_score[0]]

    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      word_ids = []
      attentions = []
      states = []
      current = end_hyp
      while current.parent is not None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        states.append(current.output.state)
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(states)), [1 for _ in word_ids]))
    return results

class CTCBestPathSearch(Serializable, SearchStrategy):
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
    one_best: Whether to output the best hyp only or all completed hyps.
    scores_proc: apply an optional operation on all scores prior to choosing the top k.
                 E.g. use with :class:`xnmt.length_normalization.EosBooster`.
  """

  yaml_tag = '!CTCBestPathSearch'
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'word'])

  @serializable_init
  def __init__(self, max_len: int = 100, len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True, scores_proc: Optional[Callable[[np.ndarray], None]] = None, vocab: Optional[Vocab] = None):
    self.max_len = max_len
    self.scores_proc = scores_proc

  def generate_output(self, translator, all_state, src_length=None, forced_trg_ids=None):
    # TODO(philip30): can only do single decoding, not batched
    if forced_trg_ids is not None and forced_trg_ids.sent_len() > self.max_len:
      logger.warning("Forced decoding with a target longer than max_len. "
                     "Increase max_len to avoid unexpected behavior.")
    completed_hyp = []
    print("trg_ids "+str(forced_trg_ids))
    prev_word = None
    for length in range(int(src_length[0]/8)):
        current_output = translator.generate_one_step(prev_word, all_state, length)
        score = current_output.logsoftmax.npvalue().transpose() # all S at current t
        self.blank_index = score.shape[0]-1
        if self.scores_proc:
          self.scores_proc(score)
        cur_word = np.argmax(score)
        completed_hyp.append(self.Hypothesis(score[cur_word],current_output,cur_word))
        prev_word = cur_word 
        print(cur_word)
    prev_word = None 
    decoded_seq = []
    final_score = 0 
    #remove duplicate chars
    for t in range(len(completed_hyp)):
      cur_word = completed_hyp[t].word 
      final_score += completed_hyp[t].score
      if prev_word != cur_word:
        decoded_seq.append(cur_word)
      prev_word = cur_word
    print("before decoding"+str(decoded_seq))
    #remove blank
    decoded_seq_final = []
    for s in decoded_seq:
      if s!=self.blank_index:
        decoded_seq_final.append(s)  
    print("after decoding"+str(decoded_seq_final))
    #remove blank

    results = []
    logsoftmaxes = []
    states = []
    results.append(CTCSearchOutput([decoded_seq_final],
                                  [final_score], list(reversed(logsoftmaxes)), list(reversed(states)), None))    
    return results

class CTCBeamSearch(Serializable, SearchStrategy):
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
    one_best: Whether to output the best hyp only or all completed hyps.
    scores_proc: apply an optional operation on all scores prior to choosing the top k.
                 E.g. use with :class:`xnmt.length_normalization.EosBooster`.
  """

  yaml_tag = '!CTCBeamSearch'
  Hypothesis = namedtuple('Hypothesis', ['score_b', 'score_nb', 'score', 'output', 'parent', 'word'])

  @serializable_init
  def __init__(self, beam_size: int = 1, max_len: int = 100, len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True, scores_proc: Optional[Callable[[np.ndarray], None]] = None, vocab: Optional[Vocab] = None):
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc
    self.end_ss =1

  def generate_output(self, translator, all_state, src_length=None, forced_trg_ids=None):
    # TODO(philip30): can only do single decoding, not batched
    if forced_trg_ids is not None and forced_trg_ids.sent_len() > self.max_len:
      logger.warning("Forced decoding with a target longer than max_len. "
                     "Increase max_len to avoid unexpected behavior.")
    active_hyp = [self.Hypothesis(0, 0, 0, None, None, None)]
    completed_hyp = []
    print("trg_ids "+str(forced_trg_ids))
    for length in range(int(src_length[0]/8)):
      # Expand hyp
      new_set = []
      for hyp in active_hyp:
        if length > 0:
          prev_word = hyp.word
        else:
          prev_word = None
        if length>=src_length[0]/8:
          break
        current_output = translator.generate_one_step(prev_word, all_state, length)
        score = current_output.logsoftmax.npvalue().transpose()
        self.blank_index = score.shape[0]-1
        if self.scores_proc:
          self.scores_proc(score)
        # Next Words
        #if forced_trg_ids is None:
        top_words = np.argpartition(score, max(-len(score),-self.beam_size))[-self.beam_size:]
        #else:
          #top_words = [forced_trg_ids[length]]
        # Queue next states
        ### Know last entry, needs to determine either copy or extend  
        for cur_word in top_words: 
          #ending with blank          
          if cur_word == self.blank_index: #copy beam
            score_b = self.len_norm.normalize_partial_topk(hyp.score, score[self.blank_index], length + 1)
            score_nb = hyp.score_nb
          else: 
            score_b = hyp.score_b
            score_nb = self.len_norm.normalize_partial_topk(hyp.score_nb, score[cur_word], length + 1)
          new_set.append(self.Hypothesis(score_b,score_nb,score_b+score_nb,\
            current_output,hyp,cur_word))          
          
          #extend
          for s in range(self.blank_index-1):
            new_top_word = (cur_word, s)
            if s == cur_word: # extend the same char and add a blank in between
              score_nb = self.len_norm.normalize_partial_topk(hyp.score_b,score[s],length+1)
            else:
              score_nb = self.len_norm.normalize_partial_topk(hyp.score,score[s],length+1)

            new_set.append(self.Hypothesis(0,score_nb,score_nb,\
              current_output,hyp,new_top_word))
          #new_score = self.len_norm.normalize_partial_topk(hyp.score, score[cur_word], length + 1)
          #new_set.append(self.Hypothesis(new_score, current_output, hyp, cur_word))
      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]
    # There is no hyp reached </s>
    if len(completed_hyp) == 0:
      completed_hyp = active_hyp
    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length[0])
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)
    if self.one_best:
      hyp_and_score = [hyp_and_score[0]]
    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      logsoftmaxes = []
      word_ids = []
      states = []
      current = end_hyp
      while current.parent is not None:
        if isinstance(current.word,tuple):
          word_ids.append(current.word[0])
          word_ids.append(current.word[1])
        else:
          word_ids.append(current.word)
        # TODO(philip30): This should probably be uncommented.
        # These 2 statements are an overhead because it is need only for reinforce and minrisk
        # Furthermore, the attentions is only needed for report.
        # We should have a global flag to indicate whether this is needed or not?
        # The global flag is modified if certain objects is instantiated.
        #logsoftmaxes.append(dy.pick(current.output.logsoftmax, current.word))
        #states.append(translator.get_nobp_state(current.output.state))
        current = current.parent
      print("before decoding"+str(word_ids))
      prev_word=word_ids[0]
      decoded_seq = [prev_word]
      for i in range(1,len(word_ids)):
        curr_word = word_ids[i]
        if prev_word == self.blank_index:
          if curr_word != self.blank_index: # ignore
            decoded_seq.append(curr_word)
        else:
          if curr_word != prev_word:
            decoded_seq.append(curr_word)
        prev_word = curr_word
      decoded_seq = list(reversed(decoded_seq))
      decoded_seq.append(self.end_ss)
      print("decoded_seq"+str(decoded_seq))            
      results.append(CTCSearchOutput([decoded_seq],
                                  [score], list(reversed(logsoftmaxes)), list(reversed(states)), None))
    #print(results)
    
    return results


class SamplingSearch(Serializable, SearchStrategy):
  """
  Performs search based on the softmax probability distribution.
  Similar to greedy searchol

  Args:
    max_len (int):
    sample_size (int):
  """

  yaml_tag = '!SamplingSearch'

  @serializable_init
  def __init__(self, max_len=100, sample_size=5):
    self.max_len = max_len
    self.sample_size = sample_size

  def generate_output(self, translator, initial_state,
                      src_length=None, forced_trg_ids=None):
    outputs = []
    for k in range(self.sample_size):
      if k == 0 and forced_trg_ids is not None:
        outputs.append(self.sample_one(translator, initial_state, forced_trg_ids))
      else:
        outputs.append(self.sample_one(translator, initial_state))
    return outputs

  # Words ids, attentions, score, logsoftmax, state
  def sample_one(self, translator, initial_state, forced_trg_ids=None):
    # Search variables
    current_words = None
    current_state = initial_state
    done = None
    # Outputs
    logsofts = []
    samples = []
    states = []
    attentions = []
    masks = []
    # Sample to the max length
    for length in range(self.max_len):
      translator_output = translator.generate_one_step(current_words, current_state)
      if forced_trg_ids is None:
        sample = translator_output.logsoftmax.tensor_value().categorical_sample_log_prob().as_numpy()
        if len(sample.shape) == 2:
          sample = sample[0]
      else:
        sample = [forced_trg[length] if forced_trg.sent_len() > length else Vocab.ES for forced_trg in forced_trg_ids]
      logsoft = dy.pick_batch(translator_output.logsoftmax, sample)
      if done is not None:
        sample = [sample[i] if not done[i] else Vocab.ES for i in range(len(done))]
        # masking for logsoftmax
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        logsoft = dy.cmult(logsoft, dy.inputTensor(mask, batched=True))
        masks.append(mask)
      # Appending output
      logsofts.append(logsoft)
      samples.append(sample)
      states.append(translator.get_nobp_state(translator_output.state))
      attentions.append(translator_output.attention)
      # Next time step
      current_words = sample
      current_state = translator_output.state
      # Check done
      done = [x == Vocab.ES for x in sample]
      # Check if we are done.
      if all(done):
        break
    # Packing output
    scores = dy.esum(logsofts).npvalue()
    masks.insert(0, [1 for _ in range(len(done))])
    samples = np.stack(samples, axis=1)
    return SearchOutput(samples, attentions, scores, logsofts, states, masks)


class MctsNode(object):
  def __init__(self, parent, prior_dist, word, attention, translator, dec_state):
    self.parent = parent
    self.prior_dist = prior_dist  # log of softmax
    self.word = word
    self.attention = attention

    self.translator = translator
    self.dec_state = dec_state

    self.tries = 0
    self.avg_value = 0.0
    self.children = {}

    # If the child is unvisited, set its avg_value to
    # parent value - reduction where reduction = c * sqrt(sum of scores of all visited children)
    # where c is 0.25 in leela
    self.reduction = 0.0

  def choose_child(self):
    return max(range(len(self.prior_dist)),
               key=lambda move: self.compute_priority(move))

  def compute_priority(self, move):
    if move not in self.children:
      child_val = self.prior_dist[move] + self.avg_value - self.reduction
      child_tries = 0
    else:
      child_val = self.prior_dist[move] + self.children[move].avg_value
      child_tries = self.children[move].tries

    K = 5.0
    exp_term = math.sqrt(1.0 * self.tries + 1.0) / (child_tries + 1)
    # TODO: This exp could be done before the prior is passed into the MctsNode
    # so it's done as a big batch
    exp_term *= K * math.exp(self.prior_dist[move])
    total_value = child_val + exp_term
    return total_value

  def expand(self):
    if self.word == Vocab.ES:
      return self

    move = self.choose_child()
    if move in self.children:
      return self.children[move].expand()
    else:
      output = self.translator.generate_one_step(move, self.dec_state)
      prior_dist = output.logsoftmax.npvalue()
      attention = output.attention

      path = []
      node = self
      while node is not None:
        path.append(node.word)
        node = node.parent
      path = ' '.join(str(word) for word in reversed(path))
      print('Creating new node:', path, '+', move)
      new_node = MctsNode(self, prior_dist, move, attention,
                          self.translator, output.state)
      self.children[move] = new_node
      return new_node

  def rollout(self, sample_func, max_len):
    prefix = []
    scores = []
    prev_word = None
    dec_state = self.dec_state

    if self.word == Vocab.ES:
      return prefix, scores

    while True:
      output = self.translator.generate_one_step(prev_word, dec_state)
      logsoftmax = output.logsoftmax.npvalue()
      attention = output.attention
      best_id = sample_func(logsoftmax)
      print("Rolling out node with word=", best_id, 'score=', logsoftmax[best_id])

      prefix.append(best_id)
      scores.append(logsoftmax[best_id])

      if best_id == Vocab.ES or len(prefix) >= max_len:
        break
      prev_word = best_id
      dec_state = output.state
    return prefix, scores

  def backup(self, result):
    print('Backing up', result)
    self.avg_value = self.avg_value * (self.tries / (self.tries + 1)) + result / (self.tries + 1)
    self.tries += 1
    if self.parent is not None:
      my_prob = self.parent.prior_dist[self.word]
      self.parent.backup(result + my_prob)

  def collect(self, words, attentions):
    if self.word is not None:
      words.append(self.word)
      attentions.append(self.attention)
    if len(self.children) > 0:
      best_child = max(self.children.itervalues(), key=lambda child: child.visits)
      best_child.collect(words, attentions)


def random_choice(logsoftmax):
  #logsoftmax *= 100
  probs = np.exp(logsoftmax)
  probs /= sum(probs)
  choices = np.random.choice(len(probs), 1, p=probs)
  return choices[0]


def greedy_choice(logsoftmax):
  return np.argmax(logsoftmax)


class MctsSearch(Serializable, SearchStrategy):
  """
  Performs search with Monte Carlo Tree Search
  """
  yaml_tag = '!MctsSearch'

  @serializable_init
  def __init__(self, visits=200, max_len=100):
    self.max_len = max_len
    self.visits = visits

  def generate_output(self, translator, dec_state, src_length=None, forced_trg_ids=None):
    assert forced_trg_ids is None
    orig_dec_state = dec_state

    output = translator.generate_one_step(None, dec_state)
    dec_state = output.state
    assert dec_state == orig_dec_state
    logsoftmax = output.logsoftmax.npvalue()
    root_node = MctsNode(None, logsoftmax, None, None, translator, dec_state)
    for i in range(self.visits):
      terminal = root_node.expand()
      words, scores = terminal.rollout(random_choice, self.max_len)
      terminal.backup(sum(scores))
      print()

    print('Final stats:')
    for word in root_node.children:
      print (word, root_node.compute_priority(word), root_node.prior_dist[word] + root_node.children[word].avg_value, root_node.children[word].tries)
    print()

    scores = []
    logsoftmaxes = []
    word_ids = []
    attentions = []
    states = []
    masks = []

    node = root_node
    while True:
      if len(node.children) == 0:
        break
      best_word = max(node.children, key=lambda word: node.children[word].tries)
      score = node.prior_dist[best_word]
      attention = node.children[best_word].attention

      scores.append(score)
      logsoftmaxes.append(node.prior_dist)
      word_ids.append(best_word)
      attentions.append(attention)
      states.append(node.dec_state)
      masks.append(1)

      node = node.children[best_word]

    word_ids = np.expand_dims(word_ids, axis=0)
    return [SearchOutput(word_ids, attentions, scores, logsoftmaxes, states, masks)]
