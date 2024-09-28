import os
import shutil

dic = {
    '公司标志': 'company_logo',
    '环保标志': 'environment_logo',
    '机器设备': 'machine_equipment',
    '奖状与证书': 'awards_paper',
    '整页图片': 'poster_picture',
    '人物': 'people',
    '人物纯黑': 'blacked_people',
    '色块': 'color_block',
    '无意义的背景图': 'meaningless_background',
}

label_dic = {
    'company_logo': 0,
    'environment_logo': 1,
    'machine_equipment': 2,
    'awards_paper': 3,
    'poster_picture': 4,
    'people': 5,
    'blacked_people': 6,
    'color_block': 7,
    'meaningless_background': 8,
}


def file_rename_and_rearrange(root_dir, origin_name, new_name):
    origin_path = os.path.join(root_dir, origin_name)
    new_path = os.path.join(root_dir, 'images/')
    idx = 0
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # 获取 origin_path 下的所有文件,如果有子文件夹,则获取子文件夹中所有文件
    for root, dirs, files in os.walk(origin_path):
        for file in files:
            # 获取文件的绝对路径
            file_path = os.path.join(root, file)
            # 将文件复制到 new_path 下, 原始的文件还是要存在
            # 并且新的文件名为 new_name_{idx}，后缀名不变
            new_file_name = new_name + '_' + str(idx) + os.path.splitext(file)[-1]
            new_file_path = os.path.join(new_path, new_file_name)
            shutil.copy2(file_path, new_file_path)
            idx += 1


def rearrange_file_by_label(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # 将每一行分割成图片路径和标签，并存储到字典中
    data_dict = {}
    for line in lines:
        file_path, label = line.strip().split()
        if label not in data_dict:
            data_dict[label] = []
        data_dict[label].append(file_path)

    # 按照标签顺序遍历字典，并写入新文件
    sorted_lines = []
    labels = sorted(data_dict.keys())
    while any(data_dict[label] for label in labels):  # 只要有一个标签还有图片，就继续
        for label in labels:
            if data_dict[label]:
                file_path = data_dict[label].pop(0)  # 取出并移除第一个元素
                sorted_lines.append(f"{file_path} {label}\n")

    # 将排序后的数据写回到新文件
    with open(output_file_path, 'w') as file:
        file.writelines(sorted_lines)


# 切分数据集
def split_dataset(root_dir, shots):
    for key in dic.keys():
        get_files = []
        origin_path = os.path.join(root_dir, key)
        for root, dirs, files in os.walk(origin_path):
            for file in files:
                get_files.append(file)

        # 输出 train_{shots}.txt, 追加而不是覆盖写入
        with open(f'../datasplit/train.txt', 'a') as f:
            idx = 0
            for file in get_files[:shots]:
                subfix = os.path.splitext(file)[-1]
                f.write(f'{dic[key]}_{idx}{subfix} {label_dic[dic[key]]}\n')
                idx += 1

        # 输出 cache_{shots}.txt
        with open(f'../datasplit/cache.txt', 'a') as f:
            idx = 0
            for file in get_files[shots:shots * 2]:
                subfix = os.path.splitext(file)[-1]
                f.write(f'{dic[key]}_{idx}{subfix} {label_dic[dic[key]]}\n')
                idx += 1

        # 输出 val_{shots}.txt
        with open(f'../datasplit/val.txt', 'a') as f:
            idx = 0
            for file in get_files[shots * 2:shots * 3]:
                subfix = os.path.splitext(file)[-1]
                f.write(f'{dic[key]}_{idx}{subfix} {label_dic[dic[key]]}\n')
                idx += 1

        # 剩下的文件输出为 test_{shots}.txt
        with open(f'../datasplit/test.txt', 'a') as f:
            idx = 0
            for file in get_files[shots * 3:]:
                subfix = os.path.splitext(file)[-1]
                f.write(f'{dic[key]}_{idx}{subfix} {label_dic[dic[key]]}\n')
                idx += 1

        print(f'{key} done!')
    rearrange_file_by_label('../datasplit/train.txt', '../datasplit/train.txt')
    rearrange_file_by_label('../datasplit/cache.txt', '../datasplit/cache.txt')
    rearrange_file_by_label('../datasplit/val.txt', '../datasplit/val.txt')
    rearrange_file_by_label('../datasplit/test.txt', '../datasplit/test.txt')


if __name__ == '__main__':
    root_dir = 'E:\DATASETS\\test_data'
    # for key in dic.keys():
    #     file_rename_and_rearrange(root_dir, key, dic[key])
    #     print(f'{key} done!')
    split_dataset(root_dir, 8)