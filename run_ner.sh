export DATA_DIR='data_dir/ner'
export BERT_BASE_DIR='vocab_file'

python run_ner.py \
 --task_name=ner \
 --do_train=true \
 --do_eval=true \
 --data_dir=$DATA_DIR/ \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
 --max_seq_length=50 \
 --train_batch_size=32 \
 --learning_rate=2e-5 \
 --num_train_epochs=3.0 \
 --output_dir=ner_output