from xnmt.events import register_xnmt_event, register_xnmt_event_sum

class GeneratorModel(object):

  def generate(self, *args, **kwargs):
    raise NotImplementedError()

  @register_xnmt_event
  def initialize_generator(self, **kwargs):
    pass

  @register_xnmt_event
  def new_epoch(self, training_regimen, num_sents):
    pass

  @register_xnmt_event
  def set_train(self, val):
    pass

  @register_xnmt_event
  def start_sent(self, src):
    pass

  def calc_loss(self, src, trg, src_mask=None, trg_mask=None):
    raise NotImplementedError()

  @register_xnmt_event_sum
  def calc_additional_loss(self, reward):
    ''' Calculate reinforce loss based on the reward
    :param reward: The default is log likelihood (-1 * calc_loss).
    '''
    return None

