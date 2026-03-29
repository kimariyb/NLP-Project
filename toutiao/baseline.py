"""
Baseline: RandomForestRegressor
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from icecream import ic
import pandas as pd


# 定义数据集路径
TRAIN_DATA_PATH = './data/train_preprocessed.csv'
STOP_WORDS_PATH = './data/stopwords.txt'

# 读取预处理后的数据
df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')

# 构建预料库
corpus = df['words'].values

# 读取停用词
with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
    stop_words = [line.strip() for line in f]

# 构建 TF-IDF 向量
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(corpus)

# 提取目标值
y = df['label']

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 实例化模型
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=10)
model.fit(x_train, y_train)

# 模型评估
y_pred = model.predict(x_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

ic(accuracy, recall, precision, f1)
