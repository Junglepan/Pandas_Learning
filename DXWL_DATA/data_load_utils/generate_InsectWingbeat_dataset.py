import csv
import os

# FaceDetection_Path = f'D:\\资料\\数据集\\数据集\\Multivariate2018_arff\\Multivariate_arff\\FaceDetection'
InsectWingbeat_Path = '..\\InsectWingbeat'

Dimensions_dict = dict(
    FaceDetection=144,
    InsectWingbeat=200,
)


def Process_Data(data_path, target_path='../'):
    '''
    步骤：
        -1.读取.arff文件
        -2.找到@data文件，以\n为分割，得到不同dimension的数据
        -3.按不同的dimension写入数据
    :return:None
    '''
    data_set_name = data_path.split('\\')[-1]  # 数据集名字
    Dimensions = Dimensions_dict[data_set_name] # 获取维度
    # 处理后的文件默认存放在当前项目下
    result_path = os.path.join(target_path, data_set_name)
    # 如果数据目录不存在，则创建
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    File_list = os.listdir(data_path)
    for file in File_list:
        if file.endswith('arff'):  # 如果是arff类型的文件
            # 判断是训练集还是测试集
            file_path = os.path.join(data_path, file)
            if 'train' in file.lower():
                train_or_test = 'train'
            elif 'test' in file.lower():
                train_or_test = 'test'
            else:
                raise Exception('在判断文件为训练集或测试集时报错！！！')

            with open(file_path, encoding="utf-8") as f:
                header = []
                for line in f:
                    if line.startswith("@attribute"):
                        header.append(line.split()[1])  # 存放数据集的 序列长度 或 维度 大小
                    elif line.startswith("@data"):
                        break
                if os.path.getsize(file_path) > 0:
                    data_f = f
                else:
                    print("---发现一个空数据文件---" + file_path)
                    continue
                # 读取数据中的每一行
                # 创建一个字典，用于存放不同维度的数据
                data_dict = {}
                for i in range(1, Dimensions + 1):
                    data_dict[i] = []
                # 用于存放标签
                label_list = []
                for data_line in data_f:  # 每一个data_line就是一个实例，即一条数据
                    # 去除单引号
                    data_line = data_line.replace("'", "").strip()
                    # 以\n为分割
                    data_label = data_line.split('\\n')  # data_label的长度为144，即维度
                    if len(data_label) != Dimensions:
                        raise Exception('数据集的维度不一致！！！')
                    label = data_label[-1].split(',')[-1]  # 获取标签
                    data_label[-1] = ','.join(data_label[-1].split(',')[:-1])  # 去掉标签和逗号
                    label_list.append([label])
                    # 将数据按不同的维度索引放到data_dict中
                    for j in range(len(data_label)):
                        data = [i for i in data_label[j].split(',')]
                        data_dict[j + 1].append(data)
                # 创建dimension文件，将数据存放到不同的Dimension文件下
                for dimension_index in data_dict.keys():
                    data = data_dict[dimension_index]
                    # 创建文件名
                    file_name = data_set_name + 'Dimension' + str(
                        dimension_index) + '_' + train_or_test.upper() + '.csv'
                    file_name_path = os.path.join(result_path, file_name)
                    with open(file_name_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        # 写入数据
                        writer.writerows(data)
                        print(f'已处理完dimension_index：{dimension_index}!!!')
                # 将标签写入到csv中
                label_name = train_or_test.lower() + '_label.csv'
                label_name_path = os.path.join(result_path, label_name)
                with open(label_name_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    # 写入数据
                    writer.writerows(label_list)


Process_Data(InsectWingbeat_Path)