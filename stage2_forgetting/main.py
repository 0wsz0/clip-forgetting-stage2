import os
import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from stage2_forgetting.utils.surgery1 import find_cat_sensitive_dimensions, apply_cat_forgetting_surgery
from stage2_forgetting.utils.data_loader import *
from stage2_forgetting.utils.visualization import *

DEVICE = "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
]
# CIFAR-10 原始标签映射：2=鸟，3=猫，5=狗
# 映射成 0,1,2 方便计算
LABEL_TO_IDX = {3:0,5:1,2:2}
IDX_TO_NAME = {0:"cat", 1:"dog", 2:"bird"}

def evaluate_model(model,images,labels,processor,surgery_dims=None,strength=0.0):
    """
    评估模型准确率  支持手术前后对比
    :param model:测试的模型
    :param images:测试的图片
    :param labels:图片的正确标签
    :param processor:CLIP的预训练的图像处理器
    :param surgery_dims:手术要操作的维度列表
    :param strength:手术强度列表
    """
    predictions = []
    # 提取文本特征（鸟/猫/狗 的 prompt）（文本特征只需计算一次，移到循环外）
    text_inputs = processor(text=PROMPTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        # 文本特征归一化（关键）
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for img in images:
            # 提取图像特征
            inputs = processor(images=img,return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            # 图像特征归一化（关键）
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # 2.如果需要手术，在这里修改特征
            if surgery_dims and strength > 0:
                image_features = apply_cat_forgetting_surgery(
                    image_features,surgery_dims,strength
                )

            # 计算相似度（矩阵乘法）
            # 图像特征[1,512] @ 文本特征 [3,512] 的转置 = [1,3]
            # 结果表示这张图跟跟“鸟/猫/狗”三个文本的匹配程度
            logits = image_features @ text_features.T
            # 【关键】乘以温度系数
            logits = logits*model.logit_scale.exp()
            # Softmax 归一化（把匹配程度变成概率）
            probs = logits.softmax(dim=1)
            # Argmax 取概率最大的为预测结果
            pred_idx = probs.argmax(dim=1).item()  # .item()把只包含一个元素的tensor张量转换成python数字
            predictions.append(pred_idx)

        # 计算准确率逻辑
        true_indices = [LABEL_TO_IDX[label] for label in labels]
        results = []
        for idx in range(3):
            class_name = IDX_TO_NAME[idx]
            # 找出所有真实标签是该类的样本
            indices = [i for i,label in enumerate(true_indices) if label==idx]  # enumerate()给true_indices加上索引值，返回（索引，值）
            # 统计预测正确的数量
            correct = sum(1 for i in indices if predictions[i]==idx)
            total = len(indices)
            acc = correct/total if total > 0 else 0
            results.append({
                "类别": class_name,
                "准确率": f"{acc*100:.2f}%",
                "正确数":correct,
                "总数":total
            })

        # 总体准确率
        total_correct = sum(1 for p,t in zip(predictions,true_indices) if p==t)
        total_samples = len(predictions)
        overall_acc = total_correct/total_samples if total_samples > 0 else 0
        results.append({
            "类别": "总体",
            "准确率": f"{overall_acc * 100:.2f}%",
            "正确数": total_correct,
            "总数": total_samples
        })

        return results,predictions

# ========== 主程序入口 ==========
def main():
    # 加载数据
    print("【步骤1】加载CIFAR-10数据...")
    images,labels,_ = load_cifar10_cat_dog_bird()

    print(f"筛选完成：共{len(images)}张图片")

    # 加载模型
    print("【步骤 2】加载 CLIP 模型...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()  # 设为评估模式

    # 定位猫特征维度
    print("【步骤 3】定位猫敏感特征维度...")

    cat_sensitive_dims = find_cat_sensitive_dimensions(
        images=images,
        labels=labels,
        model=model,
        processor=processor,
        top_k=80
    )

    # 评估手术效果
    print("\n【步骤 4】评估手术效果...")
    results_log = []
    all_predictions = []

    # 基准测试（无手术）
    baseline_results,baseline_pred = evaluate_model(model,images,labels,processor)
    results_log.append(("基准",0.0,baseline_results))
    all_predictions.append(("基准",baseline_pred))

    # 不同手术强度测试
    cat_acc_list,dog_acc_list,bird_acc_list = [],[],[]
    strengths = [0.5,1.0,1.5,2.0,2.5]
    for strength in strengths:
        print(f"\n手术强度：{strength*100:.0f}%")
        res,prediction = evaluate_model(
            model=model,
            images=images,
            labels=labels,
            processor=processor,
            surgery_dims=cat_sensitive_dims,
            strength=strength
        )
        results_log.append((f"强度{strength*100:.0f}%", strength, res))
        all_predictions.append((f"强度{strength*100:.0f}%", prediction))

        # 收集准确率用于绘图
        cat_acc_list.append(float(res[0]["准确率"].rstrip("%")))
        dog_acc_list.append(float(res[1]["准确率"].rstrip("%")))
        bird_acc_list.append(float(res[2]["准确率"].rstrip("%")))

    # 保存结果
    print("\n【步骤 5】保存结果...")
    os.makedirs("results",exist_ok=True)

    # 保存表格
    df = pd.DataFrame()
    for name, strength, res in results_log:
        row = {"实验名称": name}
        for r in res:
            row[r["类别"]] = r["准确率"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv("results/accuracy_results.csv", index=False, encoding="utf-8-sig")

    # 生成可视化图表
    print("\n【步骤 6】生成可视化图表...")

    df.to_csv("results/accuracy_results.csv", index=False, encoding="utf-8-sig")
    print("\n准确率结果表:")
    print(df)

    plot_accuracy_curve(
        strengths=[s * 100 for s in strengths],
        cat_acc=cat_acc_list,
        dog_acc=dog_acc_list,
        bird_acc=bird_acc_list,
        save_path='results/accuracy_curve.png'
    )
    print("\n✅ 所有可视化图表已保存至 results/ 文件夹")

if __name__ == "__main__":
    main()


