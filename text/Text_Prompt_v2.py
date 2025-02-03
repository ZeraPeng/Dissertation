import torch
import clip
import numpy as np


label_text_map = []
with open('/home/peng0185/Dissertation/text/ntu120_label.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())

ntu_semantic_text_map_gpt35= []
with open('/home/peng0185/Dissertation/text/ntu120_part_descriptions.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ntu_semantic_text_map_gpt35.append(temp_list)


# load clip model
def ntu_attributes(device):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('ViT-L/14@336px', device)
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
    torch.save(ntu120_semantic_feature_dict,'/home/peng0185/Dissertation/text_feature/ntu_semantic_part_feature_dict_gpt35_6part.tar')
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
    device = 'cpu'
    ntu_attributes(device)
    text_prompt()
