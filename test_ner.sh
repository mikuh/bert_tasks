export DATA_DIR='data_dir/ner'
export BERT_BASE_DIR='vocab_file'

python run_ner.py \
  --task_name=ner \
  --do_predict=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=ner_output \
  --max_seq_length=50 \
  --output_dir=ner_output