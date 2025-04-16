import yaml
import os
import ipdb

def first_check():
    # File path configuration
    root_path = "/usr1/home/s124mdg53_04/Dissertation/ASKG"  # Set your root directory path here
    label_file = os.path.join(root_path, 'data/ntu', 'ntu120_label.txt')
    askg_file = os.path.join(root_path, 'data/ntu', "classes_ASKG_ntu.yml")
    output_file = os.path.join(root_path, 'data/ntu', "classes_ASKG_ntu_checked.yml")

    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    with open(askg_file, 'r') as f:
        askg = yaml.load(f, Loader=yaml.FullLoader)
    # ipdb.set_trace()

    askg_list = []
    for i, key in enumerate(askg):
        if 'label' not in askg[key].keys():
            askg[key]['label'] = key
        for j, label in enumerate(labels):
            if key == label:
                askg[key]['idx'] = j
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(askg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

if __name__ == "__main__":
    main()