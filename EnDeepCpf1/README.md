### train
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


