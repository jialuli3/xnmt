import dynet as dy
import residual
import pyramidal
import conv_encoder
from embedder import ExpressionSequence
from translator import TrainTestInterface
from serializer import Serializable
import model_globals

class Encoder(TrainTestInterface):
  """
  A parent class representing all classes that encode inputs.
  """
  def __init__(self, model, global_train_params, input_dim):
    """
    Every encoder constructor needs to accept at least these 3 parameters 
    """
    raise NotImplementedError('__init__ must be implemented in Encoder subclasses')

  def transduce(self, sent):
    """Encode inputs into outputs.

    :param sent: The input to be encoded. This is duck-typed, so it is the appropriate input for this particular type of encoder. Frequently it will be a list of word embeddings, but it can be anything else.
    :returns: The encoded output. Frequently this will be a list of expressions representing the encoded vectors for each word.
    """
    raise NotImplementedError('transduce must be implemented in Encoder subclasses')

#  @staticmethod
#  def from_spec(encoder_spec, global_train_params, model):
#    """Create an encoder from a specification.
#
#    :param encoder_spec: Encoder-specific settings (encoders must consume all provided settings)
#    :param global_train_params: dictionary with global params such as dropout and default_layer_dim, which the encoders are free to make use of.
#    :param model: The model that we should add the parameters to
#    """
#    encoder_spec = dict(encoder_spec)
#    encoder_type = encoder_spec.pop("type")
#    encoder_spec["model"] = model
#    encoder_spec["global_train_params"] = global_train_params
#    known_encoders = [key for (key,val) in globals().items() if inspect.isclass(val) and issubclass(val, Encoder) and key not in ["BuilderEncoder","Encoder"]]
#    if encoder_type not in known_encoders and encoder_type+"Encoder" not in known_encoders:
#      raise RuntimeError("specified encoder %s is unknown, choices are: %s" 
#                         % (encoder_type,", ".join([key for (key,val) in globals().items() if inspect.isclass(val) and issubclass(val, Encoder)])))
#    encoder_class = globals().get(encoder_type, globals().get(encoder_type+"Encoder"))
#    return encoder_class(**encoder_spec)

class BuilderEncoder(Encoder):
  def transduce(self, sent):
    return self.builder.transduce(sent)

class LSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!LSTMEncoder'

  def __init__(self, input_dim=None, layers=1, hidden_dim=None, dropout=None, bidirectional=True):
    model = model_globals.get("model")
    input_dim = input_dim or model_globals.get("default_layer_dim")
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.input_dim = input_dim
    self.layers = layers
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    if bidirectional:
      self.builder = dy.BiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder)
    else:
      self.builder = dy.VanillaLSTMBuilder(layers, input_dim, hidden_dim, model)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)


class ResidualLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!ResidualLSTMEncoder'
  def __init__(self, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None, bidirectional=True):
    model = model_globals.get("model")
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    if bidirectional:
      self.builder = residual.ResidualBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    else:
      self.builder = residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class PyramidalLSTMEncoder(BuilderEncoder, Serializable):
  yaml_tag = u'!PyramidalLSTMEncoder'
  def __init__(self, input_dim=512, layers=1, hidden_dim=None, downsampling_method="skip", reduce_factor=2, dropout=None):
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim, model_globals.get("model"), dy.VanillaLSTMBuilder, downsampling_method, reduce_factor)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ConvBiRNNBuilder(BuilderEncoder, Serializable):
  yaml_tag = u'!ConvBiRNNBuilder'
  def init_builder(self, input_dim, layers, hidden_dim=None, chn_dim=3, num_filters=32, filter_size_time=3, filter_size_freq=3, stride=(2,2), dropout=None):
    model = model_globals.get("model")
    hidden_dim = hidden_dim or model_globals.get("default_layer_dim")
    dropout = dropout or model_globals.get("dropout")
    self.dropout = dropout
    self.builder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder,
                                            chn_dim, num_filters, filter_size_time, filter_size_freq, stride)
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
  
class ModularEncoder(Encoder, Serializable):
  yaml_tag = u'!ModularEncoder'
  def __init__(self, input_dim, modules):
    self.modules = modules
    
  def shared_params(self):
    return [set(["input_dim", "modules.0.input_dim"])]

  def transduce(self, sent):
    for i, module in enumerate(self.modules):
      sent = module.transduce(sent)
      if i<len(self.modules)-1:
        sent = ExpressionSequence(expr_list=sent)
    return sent

  def get_train_test_components(self):
    return self.modules
