import unittest

import dynet_config
import dynet as dy

from xnmt.attender import MlpAttender
from xnmt.bridge import CopyBridge
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
import xnmt.events
from xnmt.input_reader import PlainTextReader
from xnmt.lstm import UniLSTMSeqTransducer, BiLSTMSeqTransducer
from xnmt.loss_calculator import LossCalculator
from xnmt.mlp import MLP
from xnmt.param_collection import ParamManager
from xnmt.translator import DefaultTranslator

class TestForcedDecodingOutputs(unittest.TestCase):

  def assertItemsEqual(self, l1, l2):
    self.assertEqual(len(l1), len(l2))
    for i in range(len(l1)):
      self.assertEqual(l1[i], l2[i])

  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def assert_forced_decoding(self, sent_id):
    dy.renew_cg()
    outputs = self.model.generate_output(self.src_data[sent_id], sent_id,
                                         forced_trg_ids=self.trg_data[sent_id])
    self.assertItemsEqual(self.trg_data[sent_id], outputs[0].actions)

  def test_forced_decoding(self):
    for i in range(1):
      self.assert_forced_decoding(sent_id=i)

class TestForcedDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=self.trg_data[0],
                                      loss_calculator=LossCalculator()).value()
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    self.assertAlmostEqual(-outputs[0].score, train_loss, places=4)

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)
    self.model.initialize_generator(beam=1)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    dy.renew_cg()
    train_loss = self.model.calc_loss(src=self.src_data[0],
                                      trg=outputs[0].actions,
                                      loss_calculator=LossCalculator()).value()

    self.assertAlmostEqual(-outputs[0].score, train_loss, places=4)

class TestGreedyVsBeam(unittest.TestCase):
  """
  Test if greedy search produces same output as beam search with beam 1.
  """
  def setUp(self):
    layer_dim = 512
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.model = DefaultTranslator(
      src_reader=PlainTextReader(),
      trg_reader=PlainTextReader(),
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      trg_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=100),
      decoder=MlpSoftmaxDecoder(input_dim=layer_dim,
                                trg_embed_dim=layer_dim,
                                rnn_layer=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, decoder_input_dim=layer_dim, yaml_path="model.decoder.rnn_layer"),
                                mlp_layer=MLP(input_dim=layer_dim, hidden_dim=layer_dim, decoder_rnn_dim=layer_dim, vocab_size=100, yaml_path="model.decoder.rnn_layer"),
                                bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    self.model.set_train(False)

    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_greedy_vs_beam(self):
    dy.renew_cg()
    self.model.initialize_generator(beam=1)
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    output_score1 = outputs[0].score

    dy.renew_cg()
    self.model.initialize_generator()
    outputs = self.model.generate_output(self.src_data[0], 0,
                                         forced_trg_ids=self.trg_data[0])
    output_score2 = outputs[0].score

    self.assertAlmostEqual(output_score1, output_score2)


if __name__ == '__main__':
  unittest.main()
