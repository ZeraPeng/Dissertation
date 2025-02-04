import torch

unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split
seen_classes = list(set(range(60))-set(unseen_classes))  # ntu60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_descriptions = torch.load('text_feature/ntu_semantic_part_feature_dict_gpt35_6part.tar')

label = torch.load('text_feature/ntu_label_text_aug.tar')

label = label.to(device)
print(label.shape)

# load part language description
part_language = []
for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
    part_language.append(action_descriptions[i+1].unsqueeze(1))
part_language1 = torch.cat(part_language, dim=1).cuda(device)

print("part_language shape: ", len(part_language), len(part_language[0]), len(part_language[0][0], len(part_language[0][0][0])))
print("part_language1 shape: ", len(part_language1), len(part_language1[0]), len(part_language1[0][0]))

part_language = torch.cat([part_language1[l.item(),:,:].unsqueeze(0) for l in label], dim=0)
part_language_seen = part_language1[seen_classes]
sample_label_language = torch.cat([action_descriptions[0][l.item()].unsqueeze(0) for l in label], dim=0).cuda(device)
