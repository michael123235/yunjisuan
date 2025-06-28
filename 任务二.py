import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import pyLDAvis.gensim_models
import pyLDAvis
import warnings
import numpy as np
import seaborn as sns

# 忽略弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 加载数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 提取指定数量的新闻文本
def extract_texts(data, num_texts=10):
    texts = []
    for item in list(data.values())[:num_texts]:
        text = item["text"]
        texts.append(text)
    return texts

# 数据预处理
def preprocess_text(text):
    # 下载停用词和词形还原所需数据
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # 去除非字母字符并转换为小写
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # 分词
    words = text.split()
    # 过滤停用词和长度小于等于2的词
    words = [word for word in words if word not in stop_words and len(word) > 2]
    # 词形还原
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# 构建词典和语料库
def build_dictionary_and_corpus(processed_texts):
    dictionary = Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    return dictionary, corpus

# 训练LDA模型
def train_lda_model(corpus, dictionary, num_topics=10, passes=15):
    return LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

# 生成pyLDAvis交互图
def generate_lda_visualization(lda_model, corpus, dictionary, output_file='lda_visualization.html'):
    # 修改pyLDAvis交互图的左右布局，设置R和T的宽度比例
    lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, R=30, mds='tsne')
    pyLDAvis.save_html(lda_visualization, output_file)
    print(f"pyLDAvis交互图已保存为 {output_file}")

# 生成词云图
def generate_wordclouds(lda_model, num_topics=10):
    for i in range(num_topics):
        # 获取主题的前20个关键词及其概率
        topic_words = lda_model.show_topic(i, topn=20)
        topic_words_dict = {word: prob for word, prob in topic_words}
        # 修改词云图的大小和颜色
        wordcloud = WordCloud(width=1200, height=600, background_color='black', colormap='viridis')
        wordcloud.generate_from_frequencies(topic_words_dict)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {i}')
        plt.axis('off')
        plt.show()

# 生成热力图
def generate_heatmap(lda_model, corpus, num_topics=10):
    doc_topic_probs = []
    for doc in corpus:
        topic_dist = lda_model[doc]
        prob_dist = [0.0] * num_topics
        for topic, prob in topic_dist:
            prob_dist[topic] = prob
        doc_topic_probs.append(prob_dist)
    doc_topic_probs = np.array(doc_topic_probs)
    plt.figure(figsize=(14, 10))
    # 修改热力图的颜色
    sns.heatmap(doc_topic_probs, annot=True, cmap="coolwarm", xticklabels=[f'Topic {i}' for i in range(num_topics)])
    plt.title('Document-Topic Probability Distribution')
    plt.xlabel('Topics')
    plt.ylabel('Documents')
    plt.show()

# 调用大模型进行主题分析
def call_model(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.1,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("response", "无法解析模型返回结果")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return "请求失败"

# 分析每个主题的内容
def analyze_topics(lda_model, num_topics=10):
    for i in range(num_topics):
        topic_words = lda_model.show_topic(i, topn=10)
        topic_words_list = [word for word, prob in topic_words]
        topic_prompt = f"以下词汇代表了一个主题：{', '.join(topic_words_list)}。请详细描述这个主题可能涉及的内容："
        topic_analysis = call_model(topic_prompt)
        print(f"\n主题 {i} 分析：")
        print(topic_analysis)

if __name__ == "__main__":
    # 数据文件路径
    data_path = "gossipcop_v5_tiny_balanced.json"
    # 加载数据
    data = load_data(data_path)
    # 提取新闻文本
    texts = extract_texts(data)
    # 预处理文本
    processed_texts = [preprocess_text(text) for text in texts]
    # 构建词典和语料库
    dictionary, corpus = build_dictionary_and_corpus(processed_texts)
    # 训练LDA模型
    lda_model = train_lda_model(corpus, dictionary)
    # 生成pyLDAvis交互图
    generate_lda_visualization(lda_model, corpus, dictionary)
    # 生成词云图
    generate_wordclouds(lda_model)
    # 生成热力图
    generate_heatmap(lda_model, corpus)
    # 分析主题内容
    analyze_topics(lda_model)