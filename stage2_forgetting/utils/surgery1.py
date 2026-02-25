import torch

def find_cat_sensitive_dimensions(images,labels,model,processor,top_k=50):
    """
    找到对猫最敏感的特征维度  且狗和鸟激活度低
    :param images: 所有测试图片
    :param labels: 对应标签
    :param model: CLIP模型
    :param processor: CLIP处理器
    :param top_k: 返回最敏感的top_k个维度
    :return: 最敏感的维度索引列表
    """
    # 用来储存每张照片提取出的特征向量
    cat_features = []
    other_features = []

    # 遍历
    for img,label in zip(images,labels):
        #把img转成pytorch tensor("pt")
        inputs = processor(images=img,return_tensors="pt")

        with torch.no_grad():
            #获取图像特征向量（形状：[1,512]）
            features = model.get_image_features(**inputs)  # **是Python的字典解包
            # （关键）归一化后再比较
            features = features/features.norm(dim=-1,keepdim=True)

        if label == 3:
            cat_features.append(features)
        else:
            other_features.append(features)

    cat_features = torch.cat(cat_features,dim=0)  # [N_cat,512]
    other_features = torch.cat(other_features,dim=0)  # [N_other,512]

    # 计算每个维度的平均激活值（形状：[512]）
    # cat和other的差距越大，说明这个维度对cat识别越重要
    mean_cat = cat_features.mean(dim=0)
    mean_other = other_features.mean(dim=0)

    discriminative_score = mean_cat - mean_other
    # 找出数值最大的前 top_k 个维度的索引(.topk返回(values,indices)的元组)
    top_dims = discriminative_score.topk(top_k).indices.tolist()
    print(f"找到对猫最敏感的{top_k}个维度：{top_dims}")
    return top_dims

def apply_cat_forgetting_surgery(image_features,surgery_dims,strength=0.5):
    """
        对图像特征向量进行"猫遗忘手术"
        :param image_features: [N, 512] CLIP输出的图像特征
        :param surgery_dims: 要手术的维度索引列表
        :param strength: 手术强度 (0.0=无手术, 1.0=完全清除)
        :return: 手术后的特征向量
    """
    # 创建副本，避免修改原始数据
    modified_features = image_features.clone()
    # 遍历每一个需要手术的维度
    for dim in surgery_dims:
        # 核心公式：新值 = 原值*（1-强度）
        # if strength=0.5，则该维度减半
        modified_features[:,dim] *= (1-strength)

    return modified_features