# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：徐一恒
- **学号**：112304260142
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn（词语袋遇上爆米花袋）
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026-04-15

- **GitHub 仓库地址**：https://github.com/EHNGCCC/Bag-of-Words-Meets-Bags-of-Popcorn
- **GitHub README 地址**：

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.950000
- **Private Score**（如有）：0.950000
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
我先对影评文本进行了统一清洗，再进行分词和停用词过滤。具体包括：
- 读取 `labeledTrainData.tsv`、`unlabeledTrainData.tsv` 和 `testData.tsv`
- 清理 HTML 标签和 `<br />` 换行标记
- 将文本统一转换为小写
- 展开部分英文否定缩写，例如 `don't -> do not`
- 去除大部分非字母字符和无效符号
- 按单词切分文本
- 去除英文停用词，但保留 `not`、`no`、`never` 等否定词，避免情感信息丢失

这样处理后，文本更适合用于 Word2Vec 训练和后续分类模型建模。

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
我使用的是自己在比赛数据上训练的 Word2Vec，而不是直接加载外部预训练模型。具体做法如下：
- 使用有标签训练集和无标签训练集一起训练 Word2Vec
- 词向量维度设置为 `200`
- 训练参数中使用了窗口大小 `5`、最小词频 `3`
- 句子向量不是简单平均，而是采用 **TF-IDF 加权平均词向量**

除了 TF-IDF 加权的句向量之外，我还额外加入了少量人工特征，例如文本长度、正负情感词比例、否定词比例、感叹号和问号数量等，用来增强模型对情绪表达强度的刻画。

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
我实际尝试了三类模型：
- `Word2Vec + Logistic Regression`
- `Word2Vec + Random Forest`
- `Word2Vec + BiLSTM`

在本项目中，模型评估的核心指标不是 Accuracy，而是 Kaggle 官方使用的 **ROC-AUC**。我先在验证集上比较模型的 AUC，再决定最终方案。

实验结果表明：
- `Logistic Regression` 的验证集 AUC 为 `0.9479`
- `Random Forest` 的验证集 AUC 为 `0.9238`
- `BiLSTM` 的验证集 AUC 为 `0.9581`

因此，最终采用的是 **Word2Vec + BiLSTM** 模型，并使用其生成的 `submission_bilstm_auc.csv` 提交到 Kaggle，得到 `0.950000` 的 Public Score 和 `0.950000` 的 Private Score。

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 读取 Kaggle 提供的训练集、无标签语料和测试集  
2. 对影评文本做清洗、分词和停用词过滤  
3. 使用有标签数据和无标签数据训练 Word2Vec 模型  
4. 将每条影评表示为 TF-IDF 加权的句向量，并补充人工统计特征  
5. 分别训练 Logistic Regression 和 Random Forest，并用验证集 AUC 比较效果  
6. 在此基础上继续训练 Word2Vec + BiLSTM 模型，并同样用 AUC 评估  
7. 选择效果更好的模型生成测试集预测概率  
8. 导出 `submission.csv` 文件并提交到 Kaggle 平台  
9. 根据 Kaggle 返回的分数，对不同模型的结果进行比较分析

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
kaggle1/
├─ mould/
│  └─ 提供的参考模板与模块说明
├─ word2vec-nlp-tutorial/
│  └─ 比赛原始数据文件
├─ formal/
│  ├─ src/
│  │  └─ 项目核心源码，包括数据读取、预处理、特征工程和模型训练
│  ├─ artifacts/
│  │  └─ 训练得到的 Word2Vec 模型和经典模型文件
│  ├─ reports/
│  │  └─ 验证指标与可视化结果
│  ├─ submissions/
│  │  └─ 生成的 Kaggle 提交文件
│  ├─ main.py
│  ├─ README.md
│  └─ requirements.txt
└─ readme_机器学习实验2模板.md
```

各部分作用如下：
- `mould/`：老师提供的实验模板和参考代码片段
- `word2vec-nlp-tutorial/`：Kaggle 比赛原始数据
- `formal/src/`：正式项目代码
- `formal/artifacts/`：训练后保存的模型文件
- `formal/reports/`：实验统计结果和图表
- `formal/submissions/`：导出的 Kaggle 提交文件
- `formal/main.py`：项目主入口
