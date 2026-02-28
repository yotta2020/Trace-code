# BackdoorDefense

本项目基于 [OpenBackdoor](https://github.com/thunlp/OpenBackdoor) 开发

## 1 运行
### 1.1 环境准备
```bash
git clone git@github.com:RetrievalBackdoorDefense/BackdoorDefense.git
cd BackdoorDefense
pip install -r requirements.txt
```
### 1.2 数据集
- 从 https://drive.google.com/file/d/1dFS80oQBJ1L6XFwvrllYTDKLCXykyLkU/view?usp=sharing 下载
- 任务
    - defect
    - clone
    - refine
    - translate
- 格式: 以 defect 为例
```json
{
    "target": 1, 
    "func": "static void filter_mirror_setup(NetFilterState *nf, Error **errp)\n{\n    MirrorState *s = FILTER_MIRROR(nf);\n    Chardev *chr;\n    chr = qemu_chr_find(s->outdev);\n    if (chr == NULL) {\n        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,\n                  \"Device '%s' not found\", s->outdev);\n    qemu_chr_fe_init(&s->chr_out, chr, errp);", 
    "idx": 8
}
```

### 1.3 攻击准备
将 IST 的项目放置于 src/attackers/poisoners 目录下


### 1.4 运行后门防御
```shell
cd sh
bash run.sh
```

## 2 后门防御策略
| 缩写 | 论文标题 | 链接 |
| --- | --- | --- |
| ac | Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering | [链接](https://arxiv.org/abs/1811.03728) |
| ss | Spectral Signatures in Backdoor Attacks | [链接](https://arxiv.org/abs/1811.00636) |
| dan | Expose Backdoors on the Way: A Feature-Based Efficient Defense against Textual Backdoor Attacks | [链接](https://arxiv.org/abs/2210.07907) |
| badacts | BadActs: A Universal Backdoor Defense in the Activation Space | [链接](https://arxiv.org/abs/2405.11227) |


## 3 注意
- 目前只实现了后门防御的代码，即要运行后门防御需要训练好的后门模型（例如存放在 `datasets/defect/saved_models/IST_-1.1_0.1`）
- 需要补充后门攻击和后门模型训练的逻辑，可以对比参考：[OpenBackdoor](https://github.com/thunlp/OpenBackdoor)
- 在前期开发时请创建一个自己的分支，需要 merge 到 master 分支时最好先提交 Pull requests，让我 review 下，这样可以避免冲突。后期熟悉开发之后可以自行 merge。