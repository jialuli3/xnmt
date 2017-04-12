"""
Reads experiments descriptions in the passed configuration file
and runs them sequentially, logging outputs to files called <experimentname>.log
and <experimentname>.err.log, and reporting on final perplexity metrics.
"""

import configparser
import argparse
import sys
import encoder
import residual
import dynet as dy
import xnmt_train, xnmt_decode, xnmt_evaluate
from evaluator import BLEUEvaluator, WEREvaluator

class Tee:
  """
  Emulates a standard output or error streams. Calls to write on that stream will result
  in printing to stdout as well as logging to a file.
  """

  def __init__(self, name, indent=0, error=False):
    self.file = open(name, 'w')
    self.stdstream = sys.stderr if error else sys.stdout
    self.indent = indent
    self.error = error
    if error:
      sys.stderr = self
    else:
      sys.stdout = self

  def close(self):
    if self.error:
      sys.stderr = self.stdstream
    else:
      sys.stdout = self.stdstream
    self.file.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def write(self, data):
    self.file.write(data)
    self.stdstream.write(" " * self.indent + data)

  def flush(self):
    self.file.flush()
    self.stdstream.flush()


def get_or_none(key, dict, default_dict):
  return dict.get(key, default_dict.get(key, None))


def get_or_error(key, dict, default_dict):
  if key in dict:
    return dict[key]
  elif key in default_dict:
    return default_dict[key]
  else:
    raise RuntimeError("No value (or default value) passed for parameter {}".format(key))


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('experiments_file')
  argparser.add_argument('--dynet_mem', type=int)
  argparser.add_argument("--dynet-gpu", help="use GPU acceleration")
  train_args = argparser.parse_args()

  config = configparser.ConfigParser()
  if not config.read(train_args.experiments_file):
    raise RuntimeError("Could not read experiments config from {}".format(args.experiments_file))

  defaults = {"minibatch_size": None, "encoder_layers": 2, "decoder_layers": 2,
              "encoder_type": "BiLSTM", "run_for_epochs": 10, "eval_every": 1000,
              "batch_strategy": "src", "decoder_type": "LSTM", "decode_every": 0,
              "input_type":"word", "input_word_embed_dim":67, "output_word_emb_dim":67,
              "output_state_dim":67, "attender_hidden_dim":67, "output_mlp_hidden_dim":67,
              "encoder_hidden_dim":64, "trainer":"sgd", "eval_metrics":"bleu"}

  if "defaults" in config.sections():
    defaults.update(config["defaults"])

  del config["defaults"]

  results = []

  for experiment in config.sections():
    print("=> Running {}".format(experiment))

    output = Tee(experiment + ".log", 3)
    err_output = Tee(experiment + ".err.log", 3, error=True)
    print("> Training")

    c = config[experiment]

    encoder_type = get_or_error("encoder_type", c, defaults).lower()
    if encoder_type == "BiLSTM".lower():
      encoder_builder = encoder.BiLSTMEncoder
    elif encoder_type == "ResidualLSTM".lower():
      encoder_builder = encoder.ResidualLSTMEncoder
    elif encoder_type == "ResidualBiLSTM".lower():
      encoder_builder = encoder.ResidualBiLSTMEncoder
    elif encoder_type == "PyramidalBiLSTM".lower():
      encoder_builder = encoder.PyramidalBiLSTMEncoder
    else:
      raise RuntimeError("Unkonwn encoder type {}".format(encoder_type))

    decoder_type = get_or_error("decoder_type", c, defaults).lower()
    if decoder_type == "LSTM".lower():
      decoder_builder = dy.LSTMBuilder
    elif decoder_type == "ResidualLSTM".lower():
      decoder_builder = lambda num_layers, input_dim, hidden_dim, model:\
        residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim, model, dy.LSTMBuilder)
    else:
      raise RuntimeError("Unkonwn decoder type {}".format(encoder_type))

    # Simulate command-line arguments
    class Args: pass

    train_args = Args()
    minibatch_size = get_or_error("minibatch_size", c, defaults)
    train_args.minibatch_size = int(minibatch_size) if minibatch_size is not None else None
    train_args.eval_every = int(get_or_error("eval_every", c, defaults))
    train_args.batch_strategy = get_or_error("batch_strategy", c, defaults)
    train_args.train_source = get_or_error("train_source", c, defaults)
    train_args.train_target = get_or_error("train_target", c, defaults)
    train_args.dev_source = get_or_error("dev_source", c, defaults)
    train_args.dev_target = get_or_error("dev_target", c, defaults)
    train_args.model_file = get_or_error("model_file", c, defaults)
    train_args.input_type = get_or_error("input_type", c, defaults)
    train_args.input_word_embed_dim = int(get_or_error("input_word_embed_dim", c, defaults))
    train_args.output_word_emb_dim = int(get_or_error("output_word_emb_dim", c, defaults))
    train_args.output_state_dim = int(get_or_error("output_state_dim", c, defaults))
    train_args.attender_hidden_dim = int(get_or_error("attender_hidden_dim", c, defaults))
    train_args.output_mlp_hidden_dim = int(get_or_error("output_mlp_hidden_dim", c, defaults))
    train_args.encoder_hidden_dim = int(get_or_error("encoder_hidden_dim", c, defaults))
    train_args.trainer = get_or_error("trainer", c, defaults)

    run_for_epochs = int(get_or_error("run_for_epochs", c, defaults))
    decode_every = int(get_or_error("decode_every", c, defaults))
    test_source = get_or_error("test_source", c, defaults)
    test_target = get_or_error("test_target", c, defaults)
    temp_file_name = get_or_error("tempfile", c, defaults)

    decode_args = Args()
    decode_args.model = train_args.model_file
    decode_args.source_file = test_source
    decode_args.target_file = temp_file_name
    decode_args.input_type = get_or_error("input_type", c, defaults)

    evaluate_args = Args()
    evaluate_args.ref_file = test_target
    evaluate_args.target_file = temp_file_name
    evaluate_args.eval_metrics = get_or_error("eval_metrics", c, defaults)
    evaluators = []
    for metric in evaluate_args.eval_metrics.split(","):
      if metric == "bleu":
        evaluators.append(BLEUEvaluator(ngram=4))
      elif metric == "wer":
        evaluators.append(WEREvaluator())
      else:
        raise RuntimeError("Unkonwn evaluation metric {}".format(metric))
    xnmt_trainer = xnmt_train.XnmtTrainer(train_args,
                                          encoder_builder,
                                          int(get_or_error("encoder_layers", c, defaults)),
                                          decoder_builder,
                                          int(get_or_error("decoder_layers", c, defaults)))

    eval_score = "unknown"

    for i_epoch in xrange(run_for_epochs):
      xnmt_trainer.run_epoch()

      if decode_every != 0 and i_epoch % decode_every == 0:
        xnmt_decode.xnmt_decode(decode_args)
        eval_scores = []
        for evaluator in evaluators:
          eval_score = xnmt_evaluate.xnmt_evaluate(evaluate_args, evaluator)
          print("{}: {}".format(evaluator.metric_name(), eval_score))
          eval_scores.append(eval_score)
        # Clear the temporary file
        open(temp_file_name, 'w').close()

    results.append((experiment, eval_scores))

    output.close()
    err_output.close()

  print("")
  print("{:<20}|{:<20}".format("Experiment", "Final Scores"))
  print("-"*(20*3+2))

  for line in results:
    experiment, eval_scores = line
    print("{:<20}|{:>20}".format(experiment, eval_scores))
