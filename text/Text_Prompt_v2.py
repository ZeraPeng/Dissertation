import torch
import clip
import numpy as np


label_text_map = []
with open('/usr1/home/s124mdg53_04/Dissertation/text/ntu120_label.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())

ntu_semantic_text_map_gpt35 = []
with open('/usr1/home/s124mdg53_04/Dissertation/text/ntu120_part_descriptions.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ntu_semantic_text_map_gpt35.append(temp_list)


def ntu_label():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # clip_model, _ = clip.load('ViT-L/14@336px', device)
    clip_model, _ = clip.load('ViT-B/32', device)
    # clip_model.cuda(device)

    ntu120_label_text_dict = []
    
    for i in range(0, len(lines)):
        with torch.no_grad():
            outputs = clip_model.float().encode_text(lines[i])
            ntu120_label_text_dict.append(outputs.float().cpu())
        
    # 确保输出shape正确
    assert ntu120_label_text_dict.shape == (120, 768), f"输出shape应为(120, 768)，但得到{ntu120_label_text_dict.shape}"
    
    torch.save(ntu120_label_text_dict,'/home/peng0185/Dissertation/text_feature/ntu_label_text.tar')

def ntu_label(file_path, batch_size=32):
    """
    从文件中读取文本并使用CLIP进行编码
    
    参数:
    file_path: 文本文件路径
    batch_size: 批处理大小，用于控制内存使用
    
    返回:
    numpy数组，shape为(120, 768)
    """
    # 设置设备并加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('ViT-L/14@336px', device)
    
    # 读取文本文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 确保正好有120行
    assert len(lines) == 120, f"文件应该包含120行，但实际包含{len(lines)}行"
    
    # 预处理文本：去除空白字符
    lines = [line.strip() for line in lines]
    
    # 用于存储编码结果
    embeddings = []
    
    # 分批处理
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i + batch_size]
        
        # 使用CLIP的tokenizer处理文本
        text_tokens = clip.tokenize(batch_lines).to(device)
        
        # 获取文本特征
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            # 将特征转移到CPU并转换为numpy数组
            batch_embeddings = text_features.cpu().numpy()
            embeddings.append(batch_embeddings)
    
    # 合并所有批次的结果
    embeddings = np.concatenate(embeddings, axis=0)
    
    # 确保输出shape正确
    feature_dim = embeddings.shape[1]  # CLIP ViT-L/14 的特征维度
    assert embeddings.shape == (120, feature_dim), \
        f"输出shape应为(120, {feature_dim})，但得到{embeddings.shape}"
    
    torch.save(embeddings,'/home/peng0185/Dissertation/text_feature/ntu_label_text.tar')
    print("ntu120 label text feature saved.")

    return embeddings

# load clip model
def ntu_attributes():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # clip_model, _ = clip.load('ViT-L/14@336px', device)
    clip_model, _ = clip.load('ViT-B/32', device)
    # clip_model.cuda(device)

    ntu120_semantic_feature_dict = {}
    with torch.no_grad():
        text_dict = {}
        num_text_aug = 7   # 7
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ntu_semantic_text_map_gpt35])   # class
            elif ii == 1:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[1])) for pasta_list in ntu_semantic_text_map_gpt35])
            elif ii == 2:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[2])) for pasta_list in ntu_semantic_text_map_gpt35])
            elif ii == 3:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[3])) for pasta_list in ntu_semantic_text_map_gpt35])
            elif ii == 4:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in ntu_semantic_text_map_gpt35])
            elif ii == 5:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[5])) for pasta_list in ntu_semantic_text_map_gpt35])
            else:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] + ',' + pasta_list[6])) for pasta_list in ntu_semantic_text_map_gpt35])
            ntu120_semantic_feature_dict[ii] = clip_model.float().encode_text(text_dict[ii].to(device))

    print(len(ntu120_semantic_feature_dict))
    print(ntu120_semantic_feature_dict[0].shape)
    # torch.save(ntu120_semantic_feature_dict,'/home/peng0185/Dissertation/text_feature/ntu_semantic_part_feature_dict_gpt35_6part.tar')
    torch.save(ntu120_semantic_feature_dict,'/usr1/home/s124mdg53_04/Dissertation/text_feature/ntu_semantic_part_feature_dict_gpt35_6part_512.tar')
    return ntu120_semantic_feature_dict



def text_prompt():
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text_map])


    classes = torch.cat([v for k, v in text_dict.items()])
    print(classes.shape)
    print(classes[0].shape)  
    torch.save(classes,'/home/peng0185/Dissertation/text_feature/ntu_label_text_aug.tar')

    return classes, num_text_aug, text_dict


if __name__ == "__main__":
    # device = 'cpu'
    # label_path = '/home/peng0185/Dissertation/text/ntu120_label.txt'
    # ntu_label(label_path)
    ntu_attributes()