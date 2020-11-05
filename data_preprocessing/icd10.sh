SAB="ICD10"
CHECKPOINT_PATH=$DATA_DIR/hyperbolic/torch_states/$SAB

mkdir -p $CHECKPOINT_PATH

python3 $VIRTUAL_ENV/src/poincare/embed.py \
       -dim 256 \
       -lr 0.3 \
       -epochs 1500 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -model distance \
       -manifold poincare \
       -dset $DATA_DIR/hyperbolic/transitive_closure/$SAB.csv \
       -checkpoint $CHECKPOINT_PATH/$SAB.pth \
       -batchsize 50 \
       -eval_each 100 \
       -fresh \
       -sparse \
       -train_threads 5