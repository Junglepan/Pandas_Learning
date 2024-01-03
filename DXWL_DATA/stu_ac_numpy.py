import pandas as pd

df = pd.read_csv('dxwl.csv')
# print(df.head())
# print(df.shape)
df.sort_values(by='时间')
df_dropTeacher = df.drop(df[(df['账号'] < 20000000000)].index)   # 只保留学生的数据
df_dropTeacher.sort_values(by=['时间'])   # 排序
start_time = pd.to_datetime(df['时间'].iloc[0].split(" ")[0])   #获取时间区间的开始和结尾
end_time = pd.to_datetime(df['时间'].iloc[df['时间'].shape[0] - 1].split(" ")[0])
# print(f'{start_time}  {end_time}')
# start_time = pd.to_datetime('2022-02-20 00:00:00')
# end_time = pd.to_datetime('2022-06-26 00:00:00')

# df的时间列格式转换
df_dropTeacher['时间'] = df_dropTeacher['时间'].apply(lambda x: x.split(" ")[0])
df_dropTeacher['时间'] = pd.to_datetime(df_dropTeacher['时间'])
df_dropTeacher = df_dropTeacher.set_index('时间')
# print(df_dropTeacher)

# 区分不同的行为：学习行为和缺席
dict_activit = {'参与抢答': 0, '获得考试成绩': 1, '参与签到': 2, '学生访问章节': 3, '完成任务点': 4,
                '批阅互评作业': 5, '提交作业': 6, '提交考试': 7,
                '讨论': 8, '选人被选中': 9, '获得作业成绩': 10, '病假': 11, '签到已过期': 12, '缺勤': 13, '公假': 14,
                '事假': 15}
activit_list = list(dict_activit.keys())
activit_list.insert(0, '姓名')
activit_list.insert(0, '账号')
activit_list.insert(0, '日期')
header = activit_list
# print(header)

new_list = []
times = 0
# 每个人的统计
for [credit, name], info1 in df_dropTeacher.groupby(['账号', '姓名']):  # 根据账号和姓名区分每个同学
    #     print(f"{[credit,name]}\n")
    #     print(info1.dtypes)
    length = info1.shape[1]
    info1.loc[start_time] = info1.iloc[0].values
    info1.loc[end_time] = info1.iloc[0].values
    for time, df_W in info1.resample('W'):          # 按时间段划分：周
        #         print(f"{time}\n{df_W}")
        stu_list = [0] * 19
        stu_list[0] = time
        stu_list[1] = credit
        stu_list[2] = name.strip("\t")
        for course, info2 in df_W.groupby('事件类型'):        # 统计每周学生行为
            #             print(f'{info2}\n')
            index = dict_activit[course.strip('\t')]
            count_num = info2['事件类型'].count()
            stu_list[index + 3] = count_num
        #             print(f'{stu_list}\n')
        new_list.append(stu_list)
#     times = times+1
#     if(times == 10):
#         break

new_list
df_week = pd.DataFrame(new_list, columns=header)

# 合并一些稀疏行为
abscense = df_week.iloc[:, 14:19]
new_df_week = df_week.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 13]]
else_ac_df = df_week.iloc[:, [3, 4, 11, 12]]
ab_sum = abscense.sum(axis=1)
else_ac_sum = else_ac_df.sum(axis=1)
new_df_week.insert(10, '其他活动', else_ac_sum)
new_df_week.insert(11, '缺席', ab_sum)
# new_df_week.to_csv("stu_ac_new.csv")
# print(new_df_week.shape)
# new_df_week


import numpy as np
groups = new_df_week.groupby(['账号', '姓名']).groups  # 获取对应学生的数据索引index
# ac_list = ['参与抢答', '获得考试成绩', '参与签到', '学生访问章节', '完成任务点', '批阅互评作业', '提交作业', '提交考试', '讨论', '选人被选中', '获得作业成绩', '病假', '签到已过期', '缺勤', '公假', '事假']
num = 0
behavior_list = []
# print(groups)
for dict1 in groups:
    #     print(dict1)
    index_list = groups[dict1]
    #     print(new_df_week.iloc[index_list])
    temp_df = new_df_week.iloc[index_list, 3:]    # 获取行为索引列
    temp_df = temp_df.iloc[:20]  # 只获取前20周数据
    df_np = temp_df.values
    #     print(df_np)

    # 对每一个维度进行标准化
    np_mean = np.mean(df_np, axis=0)
    np_std = np.std(df_np, axis=0)
    z_score_normalized_np = (df_np - np_mean) / np_std
    z_score_normalized_np[np.isnan(z_score_normalized_np)] = 0
    # 对整个时间点的向量进行 Z-Score 标准化
    z_score_normalized_np = (z_score_normalized_np - np.mean(z_score_normalized_np, axis=1)[:, np.newaxis]) / np.std(
        z_score_normalized_np, axis=1)[:, np.newaxis]
    z_score_normalized_np[np.isnan(z_score_normalized_np)] = 0
    z_score_normalized_np = np.round(z_score_normalized_np, decimals=6)   # 小数处理
    #     print(z_score_normalized_np.shape)  #维度: 样本数,时间步,向量维度

    behavior_list.append(z_score_normalized_np)
#     num = num +1
#     if(num==3):
#         break
file_path = "stu_behavior_array_same.csv"
behavior_list = np.asarray(behavior_list, dtype=object)
print(behavior_list.shape)
# print(type(behavior_list))
# print((behavior_list.shape[0]))
# print(z_score_normalized_np.reshape((behavior_list.shape[0], -1)).shape)
print(behavior_list)
np.save('dxwl_dataset/behavior_array.npy', behavior_list)        # 保存在数据npy中