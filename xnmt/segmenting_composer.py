import dynet as dy

import xnmt.linear
import xnmt.embedder

from xnmt.serializer import Serializable
from xnmt.events import register_handler, handle_xnmt_event, register_xnmt_event
from xnmt.reports import Reportable
from xnmt.vocab import Vocab

class SegmentComposer(Serializable, Reportable):
  yaml_tag = u"!SegmentComposer"

  def __init__(self, encoder, transformer):
    register_handler(self)
    self.encoder = encoder
    self.transformer = transformer

  @register_xnmt_event
  def set_input_size(self, batch_size, input_len):
    pass

  @register_xnmt_event
  def next_item(self):
    pass

  def transduce(self, inputs, word=None):
    return self.transformer.transform(self.encoder, self.encoder(inputs), word)

class SegmentTransformer(Serializable):
  def transform(self, encoder, encodings, word=None):
    raise RuntimeError("Should call subclass of SegmentTransformer instead")

class TailSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!TailSegmentTransformer"
  def transform(self, encoder, encodings, word=None):
    return encoder.get_final_states()[0]._main_expr

class TailWordSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!TailWordSegmentTransformer"

  def __init__(self, yaml_context, vocab_size=1e6, embed_size=256):
    self.vocab = Vocab()
    self.lookup = yaml_context.dynet_param_collection.param_col.add_lookup_parameters((vocab_size, embed_size))

  def transform(self, encoder, encodings, word=None):
    return encoder.get_final_states()[0]._main_expr + self.lookup[self.vocab.convert(tuple(word))]

class WordOnlySegmentTransformer(TailWordSegmentTransformer):
  yaml_tag = u"!WordOnlySegmentTransformer"
  def transform(self, encoder, encodings, word=None):
    return self.lookup[self.vocab.convert(tuple(word))]

class AverageSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!AverageSegmentTransformer"
  def transform(self, encoder, encodings, word=None):
    return dy.average(encodings.as_list())

# TODO(philip30): Complete this class!
# To test, modify the segment transformer in examples/test_segmenting.yaml into this class!
class DownsamplingSegmentTransformer(SegmentTransformer):
  yaml_tag = u"!DownsamplingSegmentTransformer"

  def __init__(self, sample_size=None):
    self.sample_size = sample_size
    # TODO(philip30): Add the dynet parameters here!

  def transduce(self, encoder, inputs, word=None):
    # TODO(philip30): Complete me
    pass

class CategorySegmentTransformer(SegmentTransformer):
  yaml_tag = u"!CategorySegmentTransformer"

  def __init__(self, yaml_context, input_dim=None, category_dim=None, embed_dim=None):
    register_handler(self)
    model = yaml_context.dynet_param_collection.param_col
    self.category_output = xnmt.linear.Linear(input_dim, category_dim, model)
    self.category_embedder = xnmt.embedder.SimpleWordEmbedder(category_dim, embed_dim)
    self.train = True

  @register_xnmt_event
  def set_input_size(self, batch_size, input_len):
    self.batch_size = batch_size
    self.input_len  = input_len
    # Log likelihood buffer
    self.ll_buffer = [dy.scalarInput(0.0) for _ in range(batch_size)]
    self.counter = 0

  @handle_xnmt_event
  def on_set_train(self, train):
    self.train = train

  @register_xnmt_event
  def next_item(self):
    self.counter = (self.counter + 1) % self.batch_size

  def transform(self, encoder, encodings, word=None):
    encoding = encoder.get_final_states()[0]._main_expr
    category_logsoftmax = dy.log_softmax(self.category_output(encoding))
    if self.train:
      category = category_logsoftmax.tensor_value().categorical_sample_log_prob().as_numpy()[0]
    else:
      category = category_logsoftmax.tensor_value().argmax().as_numpy().transpose()
    # Accumulating the log likelihood for the batch
    self.ll_buffer[self.counter] += dy.pick(category_logsoftmax, category)

    return self.category_embedder.embed(category)
