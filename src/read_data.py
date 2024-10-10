import os
import json

def merge_json_files(folder_path):
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 仅处理 .json 文件
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                merged_data.extend(data)

    return merged_data

# 使用示例
# folder_path = './data'  # 你的文件夹路径
# merged_json = merge_json_files(folder_path)
# print(merged_json)