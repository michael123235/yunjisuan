import json
import requests
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置项
DATA_FILE = "gossipcop_v5_tiny_balanced.json"
API_URL = "http://localhost:11434/api/chat"
MODEL = "deepseek-r1:7b"

# 情感特征映射表（用于量化情感倾向）
SENTIMENT_MAPPING = {
    "积极": [1.0, 0.0, 0.0],  # [积极, 消极, 中性]
    "消极": [0.0, 1.0, 0.0],
    "中性": [0.0, 0.0, 1.0],
    "混合": [0.5, 0.5, 0.0]  # 特殊情况：积极+消极
}

# 主题语义映射表（用于生成主题向量）
TOPIC_MAPPING = {
    "谣言": [1.0, 0.0, 0.0, 0.0, 0.0],  # [谣言, 新闻, 评论, 娱乐, 其他]
    "新闻": [0.0, 1.0, 0.0, 0.0, 0.0],
    "评论": [0.0, 0.0, 1.0, 0.0, 0.0],
    "娱乐": [0.0, 0.0, 0.0, 1.0, 0.0],
    "其他": [0.0, 0.0, 0.0, 0.0, 1.0]
}


def read_data(file_path):
    """读取JSON数据集"""
    if not os.path.exists(file_path):
        print(f"错误：数据集文件 {file_path} 未找到！")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 移除数据统计输出
        # print(f"成功加载数据集，共 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"数据集读取失败：{e}")
        return None


def call_llm(prompt, model=MODEL):
    """调用Ollama大模型提取特征"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"API请求失败：{e}")
    except KeyError:
        print("模型响应格式错误！响应内容：\n", response.text)
    return None


def extract_features(model_response):
    """解析模型输出，提取情感和主题特征（增强版）"""
    try:
        # 尝试直接解析JSON格式
        try:
            parsed_data = json.loads(model_response)
            if isinstance(parsed_data, list):
                parsed_data = parsed_data[0]  # 处理第一个元素

            sentiment = parsed_data.get("情感特征", "")
            topic = parsed_data.get("主题语义", "")

            if not sentiment or not topic:
                raise ValueError("JSON中缺少情感或主题字段")

        except (json.JSONDecodeError, ValueError):
            # 尝试使用正则表达式匹配
            sentiment_match = re.search(r'情感特征：\s*\[(.*?)\]', model_response)
            topic_match = re.search(r'主题语义：\s*\[(.*?)\]', model_response)

            if not sentiment_match or not topic_match:
                # 尝试更宽松的匹配
                sentiment_match = re.search(r'情感特征[:：]\s*([^\n]+)', model_response)
                topic_match = re.search(r'主题语义[:：]\s*([^\n]+)', model_response)

                if not sentiment_match or not topic_match:
                    print(f"解析失败：未找到情感或主题特征。原始响应：\n{model_response}")
                    return None, None

            sentiment = sentiment_match.group(1).strip()
            topic = topic_match.group(1).strip()

        # 移除方括号（如果存在）
        sentiment = sentiment.strip("[] ")
        topic = topic.strip("[] ")

        # 映射为数值向量
        sentiment_vector = SENTIMENT_MAPPING.get(sentiment, SENTIMENT_MAPPING["中性"])
        topic_vector = TOPIC_MAPPING.get(topic, TOPIC_MAPPING["其他"])

        print(f"解析结果：情感={sentiment} → {sentiment_vector}, 主题={topic} → {topic_vector}")
        return sentiment_vector, topic_vector

    except Exception as e:
        print(f"特征提取错误：{e}")
        return None, None


def cross_modal_attention(sentiment, topic, attention_weight=0.7):
    """
    模拟跨模态注意力机制（修复版）：
    1. 使用权重矩阵将不同维度的特征投影到相同的特征空间
    2. 计算投影后特征的余弦相似度
    3. 使用注意力权重调整特征融合比例
    """
    # 输入验证
    if sentiment is None or topic is None:
        print("警告：接收到无效特征，无法计算注意力")
        return None, 0.0

    # 转换为numpy数组
    sentiment = np.array(sentiment)
    topic = np.array(topic)

    # 定义投影矩阵（将不同维度的特征投影到相同的隐藏空间）
    hidden_dim = 6  # 选择一个中间维度
    proj_sentiment = np.random.normal(0, 0.1, (len(sentiment), hidden_dim))
    proj_topic = np.random.normal(0, 0.1, (len(topic), hidden_dim))

    # 特征投影
    projected_sentiment = np.dot(sentiment, proj_sentiment)
    projected_topic = np.dot(topic, proj_topic)

    # 计算相似度（作为注意力分数）
    similarity = cosine_similarity([projected_sentiment], [projected_topic])[0][0]

    # 计算注意力权重
    attn_sentiment = np.exp(similarity) / (np.exp(similarity) + 1)
    attn_topic = 1.0 - attn_sentiment

    # 应用注意力权重（缩放原始特征而非投影后的特征）
    weighted_sentiment = sentiment * attn_sentiment
    weighted_topic = topic * attn_topic

    # 特征融合（拼接）
    fused_feature = np.concatenate([weighted_sentiment, weighted_topic])

    # L2归一化
    if np.linalg.norm(fused_feature) > 0:
        normalized_feature = fused_feature / np.linalg.norm(fused_feature)
    else:
        normalized_feature = fused_feature  # 防止除以零

    return normalized_feature, similarity


def softmax_classifier(feature, weights=None):
    """
    实现Softmax二分类器：
    1. 使用预定义权重矩阵（模拟训练好的分类器）
    2. 计算谣言和非谣言的概率分布
    """
    if feature is None:
        print("警告：接收到无效特征，无法进行分类")
        return np.array([0.5, 0.5])  # 返回均匀分布作为默认值

    # 如果未提供权重，使用随机初始化的权重矩阵
    if weights is None:
        # 权重矩阵形状：(特征维度, 类别数)
        weights = np.random.normal(0, 0.1, (len(feature), 2))

    # 线性变换
    logits = np.dot(feature, weights)

    # Softmax函数
    exp_logits = np.exp(logits - np.max(logits))  # 减去最大值以提高数值稳定性
    probabilities = exp_logits / np.sum(exp_logits)

    return probabilities


def process_samples(data):
    """处理前10个样本，执行完整流程"""
    if not isinstance(data, dict):
        print("错误：数据集需为字典格式！")
        return

    samples = list(data.items())[:10]
    print(f"\n===== 开始处理前10个样本 =====\n")

    # 初始化分类器权重（模拟训练好的模型）
    classifier_weights = np.random.normal(0, 0.1, (8, 2))

    for idx, (sample_id, item) in enumerate(samples, 1):
        # 构造特征提取提示词
        prompt = f"""请分析以下文本，输出固定格式的结果：
        情感特征：[积极/消极/中性/混合]
        主题语义：[谣言/新闻/评论/娱乐/其他]

        文本内容：{item['text']}"""

        print(f"\n--- 样本 {idx}/{10}（ID: {sample_id}）---")
        print(f"真实标签：{item['label']} | URL：{item['url']}")
        print("正在调用模型提取特征...")

        # 调用大模型提取特征
        response = call_llm(prompt)
        if not response:
            print("模型调用失败，跳过该样本！")
            continue

        # 解析特征
        sentiment, topic = extract_features(response)

        # 更严格的特征验证
        if sentiment is None or topic is None:
            print("特征提取失败，跳过该样本！")
            continue

        # 应用跨模态注意力机制
        fused_feature, attention_score = cross_modal_attention(sentiment, topic)

        if fused_feature is None:
            print("注意力计算失败，跳过该样本！")
            continue

        # 使用Softmax分类器预测
        probabilities = softmax_classifier(fused_feature, classifier_weights)
        is_rumor = probabilities[0] > probabilities[1]

        # 输出结果
        print("\n分析结果：")
        print(f"情感特征：{list(SENTIMENT_MAPPING.keys())[list(SENTIMENT_MAPPING.values()).index(sentiment)]}")
        print(f"主题语义：{list(TOPIC_MAPPING.keys())[list(TOPIC_MAPPING.values()).index(topic)]}")
        print(f"注意力分数：{attention_score:.4f}")
        print(f"分类概率：[谣言: {probabilities[0]:.4f}, 非谣言: {probabilities[1]:.4f}]")
        print(f"预测标签：{'谣言' if is_rumor else '非谣言'}")
        print(f"真实标签：{item['label']}")
        print(
            f"预测结果：{'正确' if (is_rumor and item['label'] == 'rumor') or (not is_rumor and item['label'] == 'non-rumor') else '错误'}")


def main():
    print("=== 分离式多模态分析系统启动 ===")
    data = read_data(DATA_FILE)
    if data:
        process_samples(data)
    print("\n=== 分析任务完成 ===")


if __name__ == "__main__":
    main()