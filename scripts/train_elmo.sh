export EXPERIMENT_DIR='/workspace/umls/umls_explorer/experiments/test'
export IS_BASELINE=false

CONFIG_FILE=/workspace/umls/umls_explorer/allennlp_config/elmo_hype.jsonnet
allennlp train $CONFIG_FILE --include-package elmo_hype -s $EXPERIMENT_DIR/elmo_htlm_reparam_hypeL1_0.5
