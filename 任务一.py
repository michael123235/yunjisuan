import json
import requests
import numpy as np

# 数据文件路径
DATA_FILE = "gossipcop_v5_tiny_balanced.json"

# ollama模型端点
API_URL = "http://localhost:11434/api/chat"


def read_data(file_path):
    """读取JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def call_llm(prompt, model="deepseek-r1:7b"):
    """调用ollama部署的模型"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"].strip()
    except Exception as e:
        print(f"调用模型出错: {e}")
        return None


def task1_judge_authenticity(data):
    """任务1：判断新闻真实性并统计准确率"""
    true_labels = []
    pred_labels = []
    task1_results = {}  # 存储每条新闻的判断结果

    prompt_template = """[严格格式要求] 请判断以下新闻是真实新闻(标签为1)还是虚假新闻(标签为0)，必须直接返回数字0或1，不要包含任何其他内容。

示例:
新闻内容: "科学家发现新型可再生能源技术"
返回: 1

新闻内容: "外星人将于明日访问地球"
返回: 0

当前新闻内容:
{content}

返回: """

    for item_id, item in data.items():
        true_label = 1 if item["label"] == "legitimate" else 0
        true_labels.append(true_label)

        prompt = prompt_template.format(content=item["text"])
        response = call_llm(prompt)
        pred = process_response(response)

        pred_labels.append(pred)
        task1_results[item_id] = pred  # 保存本条新闻的预测结果

    accuracy, accuracy_fake, accuracy_true = calculate_accuracy(true_labels, pred_labels)
    return {
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "accuracy": accuracy,
        "accuracy_fake": accuracy_fake,
        "accuracy_true": accuracy_true,
        "item_results": task1_results  # 返回每条新闻的判断结果
    }


def process_response(response):
    """处理模型响应，提取0或1"""
    if not response:
        return -1

    for char in response:
        if char in ['0', '1']:
            return int(char)

    if "真实" in response or "legitimate" in response or "真" in response:
        return 1
    elif "虚假" in response or "fake" in response or "假" in response:
        return 0

    return -1


def task2_analyze_sentiment(data):
    """任务2：分析新闻的语义情感"""
    sentiments = []
    task2_results = {}  # 存储每条新闻的情感分析结果

    prompt_template = """[格式要求] 请分析以下新闻的情感倾向，直接返回"正面"、"负面"或"中性"，不要包含其他内容。

新闻内容:
{content}

返回: """

    for item_id, item in data.items():
        prompt = prompt_template.format(content=item["text"])
        response = call_llm(prompt)
        sentiment = normalize_sentiment(response)

        sentiments.append(sentiment)
        task2_results[item_id] = sentiment  # 保存本条新闻的情感分析结果

    return {
        "sentiments": sentiments,
        "item_results": task2_results  # 返回每条新闻的情感分析结果
    }


def normalize_sentiment(response):
    """标准化情感分析结果"""
    if not response:
        return "中性"

    positive_words = ["正面", "积极", "好", "赞", "成功", "喜悦", "优秀"]
    negative_words = ["负面", "消极", "坏", "批评", "失败", "悲伤", "问题", "争议"]

    for word in positive_words:
        if word in response:
            return "正面"
    for word in negative_words:
        if word in response:
            return "负面"
    return "中性"


def task3_judge_with_sentiment(data, task2_results):
    """任务3：结合情感分析判断新闻真实性"""
    true_labels = []
    pred_labels = []
    task3_results = {}  # 存储每条新闻的判断结果

    prompt_template = """[严格格式要求] 请结合新闻内容和情感倾向，判断这是真实新闻(标签为1)还是虚假新闻(标签为0)，必须直接返回数字0或1。

示例:
新闻内容: "明星慈善晚会成功举办，筹集千万善款"
情感倾向: 正面
返回: 1

新闻内容: "某明星被曝偷税漏税，官方已介入调查"
情感倾向: 负面
返回: 0

当前新闻内容:
{content}
情感倾向:
{sentiment}

返回: """

    for item_id, item in data.items():
        true_label = 1 if item["label"] == "legitimate" else 0
        true_labels.append(true_label)

        sentiment = task2_results["item_results"][item_id]
        prompt = prompt_template.format(content=item["text"], sentiment=sentiment)
        response = call_llm(prompt)
        pred = process_response(response)

        pred_labels.append(pred)
        task3_results[item_id] = {  # 保存本条新闻的详细结果
            "sentiment": sentiment,
            "pred_label": pred
        }

    accuracy, accuracy_fake, accuracy_true = calculate_accuracy(true_labels, pred_labels)
    return {
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "accuracy": accuracy,
        "accuracy_fake": accuracy_fake,
        "accuracy_true": accuracy_true,
        "item_results": task3_results  # 返回每条新闻的判断结果
    }


def calculate_accuracy(true_labels, pred_labels):
    """计算三种准确率指标"""
    valid_indices = [i for i, p in enumerate(pred_labels) if p != -1]
    true_valid = [true_labels[i] for i in valid_indices]
    pred_valid = [pred_labels[i] for i in valid_indices]

    if not valid_indices:
        return 0, 0, 0

    correct = sum(t == p for t, p in zip(true_valid, pred_valid))
    accuracy = correct / len(valid_indices)

    fake_indices = [i for i, t in enumerate(true_valid) if t == 0]
    accuracy_fake = sum(true_valid[i] == pred_valid[i] for i in fake_indices) / len(fake_indices) if fake_indices else 0

    true_indices = [i for i, t in enumerate(true_valid) if t == 1]
    accuracy_true = sum(true_valid[i] == pred_valid[i] for i in true_indices) / len(true_indices) if true_indices else 0

    return accuracy, accuracy_fake, accuracy_true


def analyze_accuracy_improvement(task1_results, task3_results):
    """分析准确率提升情况"""
    improvements = {
        "accuracy": task3_results["accuracy"] - task1_results["accuracy"],
        "accuracy_fake": task3_results["accuracy_fake"] - task1_results["accuracy_fake"],
        "accuracy_true": task3_results["accuracy_true"] - task1_results["accuracy_true"]
    }
    return improvements


def main():
    # 读取数据（不再显示确认信息）
    data = read_data(DATA_FILE)

    # 任务1：判断新闻真实性
    print("\n===== 任务1：判断新闻真实性 =====")
    task1_results = task1_judge_authenticity(data)
    print(f"任务1 - 整体准确率: {task1_results['accuracy']:.4f}")
    print(f"任务1 - 假新闻准确率: {task1_results['accuracy_fake']:.4f}")
    print(f"任务1 - 真新闻准确率: {task1_results['accuracy_true']:.4f}")

    # 输出任务1每条新闻的判定结果
    print("\n任务1 - 每条新闻真实性判定结果:")
    for item_id, pred_label in task1_results["item_results"].items():
        label_text = "真实新闻" if pred_label == 1 else "虚假新闻" if pred_label == 0 else "无法判定"
        print(f"新闻ID: {item_id}, 预测标签: {pred_label}, 判定结果: {label_text}")

    # 任务2：分析语义情感
    print("\n===== 任务2：分析语义情感 =====")
    task2_results = task2_analyze_sentiment(data)

    # 计算情感占比
    total = len(task2_results["sentiments"])
    sentiment_stats = {
        "正面": task2_results["sentiments"].count("正面") / total * 100,
        "负面": task2_results["sentiments"].count("负面") / total * 100,
        "中性": task2_results["sentiments"].count("中性") / total * 100
    }
    print("任务2 - 情感分析占比:")
    for sentiment, ratio in sentiment_stats.items():
        print(f"{sentiment}: {ratio:.2f}%")

    # 输出任务2每条新闻的情感分析
    print("\n任务2 - 每条新闻情感分析结果:")
    for item_id, sentiment in task2_results["item_results"].items():
        print(f"新闻ID: {item_id}, 情感倾向: {sentiment}")

    # 任务3：结合情感分析判断新闻真实性
    print("\n===== 任务3：结合情感分析判断新闻真实性 =====")
    task3_results = task3_judge_with_sentiment(data, task2_results)
    print(f"任务3 - 整体准确率: {task3_results['accuracy']:.4f}")
    print(f"任务3 - 假新闻准确率: {task3_results['accuracy_fake']:.4f}")
    print(f"任务3 - 真新闻准确率: {task3_results['accuracy_true']:.4f}")

    # 输出任务3每条新闻的判断结果
    print("\n任务3 - 每条新闻结合情感分析的判定结果:")
    for item_id, result in task3_results["item_results"].items():
        label_text = "真实新闻" if result["pred_label"] == 1 else "虚假新闻" if result["pred_label"] == 0 else "无法判定"
        print(
            f"新闻ID: {item_id}, 情感倾向: {result['sentiment']}, 预测标签: {result['pred_label']}, 判定结果: {label_text}")

    # 分析准确率提升
    print("\n===== 准确率提升分析 =====")
    improvements = analyze_accuracy_improvement(task1_results, task3_results)
    for metric, improvement in improvements.items():
        status = "提升" if improvement > 0 else "下降" if improvement < 0 else "不变"
        print(f"{metric} {status}: {improvement:.4f}")


if __name__ == "__main__":
    main()