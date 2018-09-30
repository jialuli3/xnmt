!Experiment
evaluate:
- !AccuracyEvalTask
  desc: null
  eval_metrics: wer,cer
  hyp_file: '{EXP_DIR}/logs/{EXP}.test_hyp'
  inference: !AutoRegressiveInference
    batcher: !InOrderBatcher {batch_size: 1, pad_src_to_multiple: 8, src_pad_token: null,
      trg_pad_token: 1}
    max_num_sents: null
    max_src_len: 1500
    mode: onebest
    post_process: join-char
    ref_file: null
    reporter: null
    search_strategy: !BeamSearch
      beam_size: 20
      len_norm: !PolynomialNormalization {apply_during_search: true, m: 1.5}
      max_len: 500
      one_best: true
      scores_proc: null
    src_file: null
    trg_file: null
  model: !Ref {default: 1928437192847, name: null, path: model}
  ref_file: '{DATA_DIR}/test.words'
  src_file: '{DATA_DIR}/test.h5'
- !LossEvalTask
  batcher: !Ref {default: 1928437192847, name: null, path: train.batcher}
  desc: null
  loss_calculator: !AutoRegressiveKMeansLoss {dev_evaluate: false, test_evaluate: true,
    truncate_dec_batches: false}
  loss_comb_method: sum
  max_src_len: 1500
  max_trg_len: null
  model: !Ref {default: 1928437192847, name: null, path: model}
  ref_file: '{DATA_DIR}/test.char'
  src_file: '{DATA_DIR}/test.h5'
exp_global: !ExpGlobal
  bias_init: !ZeroInitializer {}
  commandline_args: &id001
    dynet_autobatch: null
    dynet_devices: null
    dynet_gpu: true
    dynet_gpu_ids: null
    dynet_gpus: null
    dynet_mem: null
    dynet_profiling: null
    dynet_seed: null
    dynet_viz: false
    dynet_weight_decay: null
    experiment_name: []
    experiments_file: las-timit-kmeans.yaml
    generate_doc: false
    settings: debug
  compute_report: false
  default_layer_dim: 512
  dropout: 0.3
  log_file: '{EXP_DIR}/logs/{EXP}.log'
  loss_comb_method: sum
  model_file: '{EXP_DIR}/models/{EXP}.mod'
  param_init: !GlorotInitializer {gain: 1.0}
  placeholders: {DATA_DIR: /home/jialu/TIMIT}
  save_num_checkpoints: 1
  truncate_dec_batches: false
  weight_noise: 0.0
model: !DefaultKMeansTranslator
  attender: !MlpAttender
    bias_init: !Ref {default: 1928437192847, name: null, path: exp_global.bias_init}
    hidden_dim: 128
    input_dim: 512
    param_init: !Ref {default: 1928437192847, name: null, path: exp_global.param_init}
    state_dim: 512
    truncate_dec_batches: false
    xnmt_subcol_name: MlpAttender.d853df2e
  calc_attention_entropy: false
  calc_global_fertility: false
  cluster: !KMeans
    max_iter: 20
    n_components: 100
    n_dims: 512
    param_init: !NormalInitializer {mean: 0, var: 1}
    xnmt_subcol_name: KMeans.7627873d
  compute_report: false
  decoder: !AutoRegressiveDecoder
    bridge: !CopyBridge {dec_dim: 512, dec_layers: 1}
    input_dim: 512
    input_feeding: true
    rnn: !UniLSTMSeqTransducer
      bias_init: !Ref {default: 1928437192847, name: null, path: exp_global.bias_init}
      decoder_input_dim: 512
      decoder_input_feeding: true
      dropout: 0.3
      hidden_dim: 512
      input_dim: 64
      layers: 1
      param_init: !Ref {default: 1928437192847, name: null, path: exp_global.param_init}
      weightnoise_std: 0.0
      xnmt_subcol_name: UniLSTMSeqTransducer.02ba6927
    scorer: !Softmax
      bias_init: !Ref {default: 1928437192847, name: null, path: exp_global.bias_init}
      input_dim: 512
      label_smoothing: 0.1
      output_projector: !Linear
        bias: true
        bias_init: !Ref {default: 1928437192847, name: null, path: exp_global.bias_init}
        input_dim: 512
        output_dim: 55
        param_init: !Ref {default: 1928437192847, name: null, path: exp_global.param_init}
        xnmt_subcol_name: Linear.99fb15e0
      param_init: !Ref {default: 1928437192847, name: null, path: exp_global.param_init}
      trg_reader: !Ref {default: 1928437192847, name: null, path: model.trg_reader}
      vocab: null
      vocab_size: null
      xnmt_subcol_name: Softmax.017f8241
    transform: !AuxNonLinear
      activation: tanh
      aux_input_dim: 512
      bias: true
      bias_init: !Ref {default: 1928437192847, name: null, path: exp_global.bias_init}
      input_dim: 512
      output_dim: 512
      param_init: !Ref {default: 1928437192847, name: null, path: exp_global.param_init}
      xnmt_subcol_name: AuxNonLinear.8c2e224a
    trg_embed_dim: 64
    truncate_dec_batches: false
    xnmt_subcol_name: AutoRegressiveDecoder.e00a487f
  encoder: !ModularSeqTransducer
    input_dim: 40
    modules:
    - !PyramidalLSTMSeqTransducer
      builder_layers:
      - - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 40
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.096311a3
        - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 40
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.e7a1907f
      - - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 1024
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.09df28a6
        - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 1024
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.0f297013
      - - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 1024
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.3fff25cf
        - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 1024
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.542ceaf9
      - - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 1024
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.65b72ad5
        - !UniLSTMSeqTransducer
          bias_init: !ZeroInitializer {}
          decoder_input_dim: null
          decoder_input_feeding: true
          dropout: 0.3
          hidden_dim: 256.0
          input_dim: 1024
          layers: 1
          param_init: !GlorotInitializer {gain: 1.0}
          weightnoise_std: 0.0
          xnmt_subcol_name: UniLSTMSeqTransducer.33b196c1
      downsampling_method: concat
      dropout: 0.3
      hidden_dim: 512
      input_dim: 40
      layers: 4
      reduce_factor: 2
  inference: !AutoRegressiveInference
    batcher: !InOrderBatcher {batch_size: 1, pad_src_to_multiple: 1, src_pad_token: 1,
      trg_pad_token: 1}
    max_num_sents: null
    max_src_len: null
    mode: onebest
    post_process: !PlainTextOutputProcessor {}
    ref_file: null
    reporter: null
    search_strategy: !BeamSearch
      beam_size: 1
      len_norm: !NoNormalization {}
      max_len: 100
      one_best: true
      scores_proc: null
    src_file: null
    trg_file: null
  search_strategy: !BeamSearch
    beam_size: 1
    len_norm: !NoNormalization {}
    max_len: 100
    one_best: true
    scores_proc: null
  src_embedder: !NoopEmbedder {emb_dim: 40}
  src_reader: !H5Reader {feat_from: null, feat_skip: null, feat_to: null, timestep_skip: null,
    timestep_truncate: null, transpose: true}
  trg_embedder: !SimpleWordEmbedder
    emb_dim: 64
    fix_norm: 1
    param_init: !Ref {default: 1928437192847, name: null, path: exp_global.param_init}
    src_reader: !Ref {default: 1928437192847, name: null, path: model.src_reader}
    trg_reader: !Ref {default: 1928437192847, name: null, path: model.trg_reader}
    vocab: null
    vocab_size: 55
    weight_noise: 0.0
    word_dropout: 0.1
    xnmt_subcol_name: SimpleWordEmbedder.d12237ed
  trg_reader: !PlainTextReader
    read_sent_len: false
    vocab: !Vocab
      i2w: [<s>, </s>, __, a, "\xE4", "\xE1", "\xE0", "\xE2", "\xE6", b, c, "\xE7",
        d, e, "\xE9", "\xE8", "\xEA", "\xEB", f, g, h, i, "\xED", "\xEE", "\xEF",
        j, k, l, m, n, "\xF1", o, "\xF3", "\xF6", "\xF4", "\u0153", p, q, r, s, t,
        u, "\xFA", "\xFC", "\xF9", "\xFB", v, w, x, y, "\xFF", z, "\xDF", '''', <unk>]
      sentencepiece_vocab: false
      vocab_file: null
preproc: null
random_search_report: null
train: !SimpleTrainingRegimen
  batcher: !WordSrcBatcher {avg_batch_size: 210, break_ties_randomly: true, pad_src_to_multiple: 8,
    src_pad_token: null, trg_pad_token: 1, words_per_batch: null}
  commandline_args: *id001
  dev_combinator: null
  dev_every: 0
  dev_tasks:
  - !AccuracyEvalTask
    desc: null
    eval_metrics: wer,cer
    hyp_file: '{EXP_DIR}/logs/{EXP}.dev_hyp'
    inference: !AutoRegressiveInference
      batcher: !InOrderBatcher {batch_size: 1, pad_src_to_multiple: 8, src_pad_token: null,
        trg_pad_token: 1}
      max_num_sents: null
      max_src_len: 1500
      mode: onebest
      post_process: join-char
      ref_file: null
      reporter: null
      search_strategy: !BeamSearch
        beam_size: 20
        len_norm: !PolynomialNormalization {apply_during_search: true, m: 1.5}
        max_len: 500
        one_best: true
        scores_proc: null
      src_file: null
      trg_file: null
    model: !Ref {default: 1928437192847, name: null, path: model}
    ref_file: '{DATA_DIR}/dev.words'
    src_file: '{DATA_DIR}/dev.h5'
  - !LossEvalTask
    batcher: !Ref {default: 1928437192847, name: null, path: train.batcher}
    desc: null
    loss_calculator: !AutoRegressiveKMeansLoss {dev_evaluate: true, test_evaluate: false,
      truncate_dec_batches: false}
    loss_comb_method: sum
    max_src_len: 1500
    max_trg_len: null
    model: !Ref {default: 1928437192847, name: null, path: model}
    ref_file: '{DATA_DIR}/dev.char'
    src_file: '{DATA_DIR}/dev.h5'
  dev_zero: false
  initial_patience: 15
  loss_calculator: !AutoRegressiveKMeansLoss {dev_evaluate: false, test_evaluate: false,
    truncate_dec_batches: false}
  loss_comb_method: sum
  lr_decay: 0.5
  lr_decay_times: 3
  max_num_train_sents: null
  max_src_len: 1500
  max_trg_len: 350
  model: !Ref {default: 1928437192847, name: null, path: model}
  name: '{EXP}'
  patience: 8
  reload_command: null
  restart_trainer: true
  run_for_epochs: 50
  sample_train_sents: null
  src_file: '{DATA_DIR}/train.h5'
  trainer: !AdamTrainer {alpha: 0.0003, beta_1: 0.9, beta_2: 0.999, eps: 1.0e-08,
    skip_noisy: false, update_every: 1}
  trg_file: '{DATA_DIR}/train.char'
  update_every: 1
