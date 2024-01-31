### train
```
python train.py -h
usage: train.py [-h] --train_file TRAIN_FILE --val_file VAL_FILE --output_file OUTPUT_FILE
                --model_pth MODEL_PTH [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS] [--lr LR]
                [--early_stop EARLY_STOP]

options:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        train datas which must contain "seq" and "indel_freq" two columns
  --val_file VAL_FILE   validate datas which must contain "seq" and "indel_freq" two columns
  --output_file OUTPUT_FILE
                        predicted results of validata datas
  --model_pth MODEL_PTH
                        trained model weights
  --batch_size BATCH_SIZE
                        batch size
  --n_epochs N_EPOCHS   max training epochs
  --lr LR               learning rate
  --early_stop EARLY_STOP
                        early stop patience for model no longer getting better
```

example:
```
python train.py \
--train_file dataset\EnDeepCpf1_train.csv \
--val_file dataset\EnDeepCpf1_test.csv \
--output_file test\output1.csv \
--model_pth test\trained_model.pth \
--batch_size 64 \
--n_epochs 1000 \
--lr 1e-4 \
--early_stop 10
```


### finetune
```
python finetune.py -h
usage: finetune.py [-h] --train_file TRAIN_FILE --val_file VAL_FILE --output_file OUTPUT_FILE
                   --pretrained_model_pth PRETRAINED_MODEL_PTH --finetuned_model_pth
                   FINETUNED_MODEL_PTH [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS] [--lr LR]
                   [--early_stop EARLY_STOP]

options:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        train datas which must contain "seq", "chromatin_accessibility" and
                        "indel_freq" three columns
  --val_file VAL_FILE   validate datas which must contain "seq", "chromatin_accessibility" and
                        "indel_freq" three columns
  --output_file OUTPUT_FILE
                        predicted results of validata datas
  --pretrained_model_pth PRETRAINED_MODEL_PTH
                        pretrained model weights
  --finetuned_model_pth FINETUNED_MODEL_PTH
                        finetuned model weights
  --batch_size BATCH_SIZE
                        batch size
  --n_epochs N_EPOCHS   max training epochs
  --lr LR               learning rate
  --early_stop EARLY_STOP
                        early stop patience for model no longer getting better

```

example:
```
python finetune.py \
--train_file dataset\HEK_lenti.csv \
--val_file dataset\HEK_HCT_plasmid.csv \
--output_file test\output2.csv \
--pretrained_model_pth test\trained_model.pth \
--finetuned_model_pth test\finetuned_model.pth \
--batch_size 64 \
--n_epochs 1000 \
--lr 1e-4 \
--early_stop 10
```

### predict 
```
python predict.py -h
usage: predict.py [-h] --input_file INPUT_FILE --output_file OUTPUT_FILE --model_pth MODEL_PTH
                  [--batch_size BATCH_SIZE] [--use_ca]

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        datas for predicting
  --output_file OUTPUT_FILE
                        predicted results
  --model_pth MODEL_PTH
                        trained model weights
  --batch_size BATCH_SIZE
                        batch size
  --use_ca              using chromatin accessibility information
```

example:
```
# use_ca
python predict.py \
--input_file dataset\HEK_HCT_plasmid.csv \
--output_file output\tt2.csv \
--model_pth trained_model\HEK_lenti_finetuned_model.pth \
--use_ca

# without ca
python predict.py \
--input_file dataset\EnDeepCpf1_test.csv \
--output_file output\tt1.csv \
--model_pth trained_model\EnDeepCpf1_trained_model.pth
```


