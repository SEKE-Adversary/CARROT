# CARROT

*Towards Robustness of Deep Program Processing Models -- Detection, Estimation and Enhancement*

See the [paper](https://dl.acm.org/doi/abs/10.1145/3511887) here.

Watch the [video](https://www.youtube.com/watch?v=qQzFGbbxKeA) here.

## Requirement

```
torch == 1.8.0
dgl == 0.7.2
transformers == 3.3
```

This is the recommended environment. Other versions may be compatible.

## Preparing the Dataset

**Use the pre-processed datasets**

1. Download the already pre-processed datasets -- [OJ](https://drive.google.com/drive/folders/1__SjuEKH8Sa_OYWhegiGE6Brbr1ObZrM?usp=sharing), [OJClone](https://drive.google.com/drive/folders/1PaqKmUqV-TPwSGWvUEhALW20crcfeU5D?usp=sharing), and [CodeChef](https://drive.google.com/drive/folders/1ZEIb35PzfD2ojWr53Qa_myFRMVK7QI7f?usp=sharing).

2. Put the contents of OJ, OJClone and CodeChef in the corresponding directories of `data`, `data_clone`, and `data_defect` respectively.

3. The pre-processed datasets for GRU, LSTM, ASTNN, LSCNN, TBCNN, CodeBERT and CDLH are all included in the directories now.

**Pre-process the datasets by yourself**

1. Download the raw datasets, *i.e.*, `oj.tar.gz` from [OJ](https://drive.google.com/drive/folders/1__SjuEKH8Sa_OYWhegiGE6Brbr1ObZrM?usp=sharing) and `codechef.zip` from [CodeChef](https://drive.google.com/drive/folders/1ZEIb35PzfD2ojWr53Qa_myFRMVK7QI7f?usp=sharing)

2. Put `oj.tar.gz` in the directory of `data` and `codechef.zip` in `data_defect`.

3. Run the following commands to build the OJ dataset for the DL models. The dataset format of CodeChef is almost identical to OJ, and the code can be reused.

```sh
> cd preprocess-lstm
> python3 main.py 
> cd ../preprocess-astnn
> python3 pipeline.py
> cd ../preprocess-tbcnn
> python3 main.py
> cd ..
```

4. Copy `oj.pkl.gz`, `oj_uid.pkl.gz`, and `oj_inspos.pkl.gz` in the directory of `data` and paste them into `data_clone`.

5. Run the following commands to build the OJClone dataset for the DL models.

```sh
> cd preprecess_clone-lstm
> python3 main.py
> cd ../preprocess_clone-astnn
> python3 main.py
> cd ../preprocess_clone-tbcnn
> python3 main.py
> cd ..
```

6. Everything is ready now.

## Training the DL Model

The source code directories are named according to the dataset and the model. `code`, `code_clone` and `code_defect` refers to OJ, OJClone and CodeChef, respectively.

The source code files to train each model (*i.e.*, GRU, LSTM, ASTNN, LSCNN, TBCNN, CodeBERT and CDLH) on each dataset (*i.e.*, OJ, OJClone, and CodeChef) are included in each corresponding directory. For instance, `code_defect-codebert` refers to CodeBERT for CodeChef. Note that the GRU and LSTM models are both in the directory of `lstm`.

*E.g.*, run the following commands to train a GRU model on OJ.

```sh
> cd code-lstm
> python3 lstm_train.py -gpu 0 -model GRU -lr 1e-3 -save_dir MODEL/SAVE/PATH --data ../data/oj.pkl.gz
> cd ..
```

Run the following commands to train a LSTM model on CodeChef.

```sh
> cd code_defect-lstm
> python3 lstm_train.py -gpu 0 -model LSTM -lr 1e-3 -save_dir MODEL/SAVE/PATH --data ../data_defect/oj.pkl.gz
> cd ..
```

Run the following commands to train a CDLH model on OJClone.

```sh
> cd code_clone-cdlh
> python3 train.py --save_dir MODEL/SAVE/PATH
> cd ..
```

## Adversarial Attack

Run `python3 attacker.py` in each directory to attack the DL models.

*E.g.*, run the following commands to attack the CodeBERT model on OJ.

```sh
> cd code-codebert
> python3 attacker.py --model_dir FINETUNED/CODEBERT/MODEL/PATH
> cd ..
```

The corresponding relationship between the attacking algorithm and the `Attacker` class is as the following table.

|Attacking Algorithm|Class Name|
|-|-|
|I-CARROT|Attacker|
|S-CARROT|InsAttacker|
|I-RW|AttackerRandom|
|S-RW|InsAttackerRandom|

One may use different attacking algorithm (including I-CARROT, S-CARROT, I-RW, and S-RW) by employing different `Attacker`'s in the code.

## Robustness Measurement

After adversarial attack, the logging files are obtained. Run the following commands to compute the robustness of the DL model.

```sh
> python3 compute_robustness.py -I PATH/TO/ICARROT/LOG -S PATH/TO/SCARROT/LOG
```

## Adversarial Training

Take LSTM on OJClone for example.

1. Run the following commands to create the adversarial example training set.

```sh
> cd code_clone-lstm
> python3 attacker4training.py
> cd ..
```

2. Run the following commands to adversarially train the model.

```sh
> cd code_clone-lstm
> python3 lstm_train.py --adv_train_path PATH/TO/ADVERSARIAL/EXAMPLE/SET --OTHER_ARGUMENTS
> ..
```

3. Go back to step 1 to iteratively update the adversarial example set upon the current training set.

## Citation

```bib
@article{zhang2022towards,
  title={Towards Robustness of Deep Program Processing Models--Detection, Estimation and Enhancement},
  author={Zhang, Huangzhao and Fu, Zhiyi and Li, Ge and Ma, Lei and Zhao, Zhehao and Yang, Hua’an and Sun, Yizhe and Liu, Yang and Jin, Zhi},
  journal={ACM Transactions on Software Engineering and Methodology},
  year={2022},
  publisher={ACM New York, NY}
}
```
