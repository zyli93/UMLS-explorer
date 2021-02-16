local NUM_GPUS = 1;
local NUM_GRAD_ACC = 4;
local BATCH_SIZE = 512 / NUM_GPUS / NUM_GRAD_ACC;

local HYPER_DIM = 256;
local EUCLID_DIM = 512;

local EXPERIMENT_DATA_DIR = std.extVar("EXPERIMENT_DIR") + "/data";

local BASE_READER = {
        "type": "umls",
        "tokenizer": {
          "type": "just_spaces",
        },
        "token_indexers": {
          "tokens": {
            "type": "single_id",
            "namespace": "euclidean"
          },
          "token_characters": {
            "type": "elmo_characters"
          }
        },
        "hyperbolic_phrase_indexers": {
          "tokens": {
            "type": "single_id",
            "namespace": "hyperbolic"
          }
        },
        "max_sequence_length": 400,
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"]
};

local BASE_LOADER = {
  "max_instances_in_memory": BATCH_SIZE * 100,
  "batch_sampler": {
    "type": "bucket",
    "batch_size": BATCH_SIZE,
  }
};

{
  "dataset_reader": BASE_READER,
  // Note: We don't set a validation_data_path because the softmax is only
  // sampled during training. Not sampling on GPUs results in a certain OOM
  // given our large vocabulary. We'll need to evaluate against the test set
  // (when we'll want a full softmax) with the CPU.
  "train_data_path": EXPERIMENT_DATA_DIR,

  "vocabulary": {
      // Use a prespecified vocabulary for efficiency.
      "type": "from_files",
      "directory": EXPERIMENT_DATA_DIR + "/vocab"
      // Plausible config for generating the vocabulary.
      // "tokens_to_add": {
      //     "tokens": ["<S>", "</S>"],
      //     "token_characters": ["<>/S"]
      // },
      // "min_count": {"tokens": 3}
  },
  "model": {
    "type": "hyperbolic_tuned_language_model",
    "bidirectional": true,
    "num_samples": 8192,
    # Sparse embeddings don't work with DistributedDataParallel.
    "sparse_embeddings": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "empty"
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                // Same as the Transformer ELMo in Calypso. Matt reports that
                // this matches the original LSTM ELMo as well.
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]],
                "num_highway": 2,
                "projection_dim": EUCLID_DIM,
                "projection_location": "after_highway",
                "do_layer_norm": true
            }
        }
      }
    },
    // TODO(brendanr): Consider the following.
    // remove_bos_eos: true,
    // Applies to the contextualized embeddings.
    "dropout": 0.1,
    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": EUCLID_DIM,
        "hidden_dim": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "input_dropout": 0.1
    },
    "hyperbolic_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "trainable": false,
          "pretrained_file": EXPERIMENT_DATA_DIR + "/hyperbolic/embeddings/ICD10_embedding.h5",
          "embedding_dim": HYPER_DIM,
          "vocab_namespace": "hyperbolic"
        }
      }
    },
    "hyperbolic_encoder": {
      "type": "lstm",
      "input_size": EUCLID_DIM,
      "hidden_size": HYPER_DIM
    },
    "hyperbolic_weight": 0.1
  },
  "data_loader": BASE_LOADER,
  // "distributed": {
  //   "cuda_devices": if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
  // },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      // The gradient accumulators in Adam for the running stdev and mean for
      // words not used in the sampled softmax would be decayed to zero with the
      // standard "adam" optimizer.
      "type": "dense_sparse_adam"
    },
    // TODO(brendanr): Needed with transformer too?
    // "grad_norm": 10.0,
    "learning_rate_scheduler": {
      "type": "noam",
      // See https://github.com/allenai/calypso/blob/master/calypso/train.py#L401
      "model_size": 512,
      // See https://github.com/allenai/calypso/blob/master/bin/train_transformer_lm1b.py#L51.
      // Adjusted based on our sample size relative to Calypso's.
      "warmup_steps": 6000
    },
    "num_gradient_accumulation_steps": NUM_GRAD_ACC,
    "use_amp": true
  }
}