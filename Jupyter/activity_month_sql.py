import pandas as pd
df = pd.read_csv('dxwl.csv')
# print(df.head())
# print(df.shape)
df.sort_values(by='时间')
df_dropTeacher = df.drop(df[(df['账号'] < 20000000000)].index)

# 获取时间开始和结尾日期
start_time = pd.to_datetime(df['时间'].iloc[0].split(" ")[0])
end_time = pd.to_datetime(df['时间'].iloc[df['时间'].shape[0] - 1].split(" ")[0])
# print(start_time)
# print(end_time)

# 每个月的时间序列
date_range1 = pd.date_range(start=start_time, end=end_time, freq="M")
# print(date_range1)

# df日期列的时间格式转换
df_dropTeacher['时间'] = pd.to_datetime(df_dropTeacher['时间'])
df_list = []

# 根据时间段处理
for i in range(len(date_range1)):
    if (i == 0):
        df_M = df_dropTeacher[df_dropTeacher['时间'] < date_range1[i]]
    else:
        df_M = df_dropTeacher[(df_dropTeacher['时间'] < date_range1[i]) &
                              (df_dropTeacher['时间'] > date_range1[i - 1])]

    #     print(df_M.shape)
    stu_number = df_M['姓名'].nunique()  # 学习人数
    #     print(stu_number)
    # 平均活动次数:
    ave_activity = df_M.shape[0] / stu_number
    #     print("平均活动次数:" + str(int(ave_activity)))

    # 获取每个人的活动次数 活跃人数  活跃率
    date = date_range1[i]
    ac_number = 0
    huoyue_number = 0
    huoyue_ratio = 0
    ac_sum = df_M.shape[0]
    times = 0;
    for name, info in df_M.groupby('姓名'):
        # 第i月的互动人数
        ac_number = ac_number + 1
        # 第i月活跃人数
        #         print(f'{name}\n{info}\n')
        if (info['事件类型'].count() >= ave_activity):
            huoyue_number = huoyue_number + 1
        huoyue_ratio = format(huoyue_number / ac_number, '.2f')

    #          if(times >10):
    #             break
    #     print('第{}月的互动人数:{}'.format(i,ac_number))
    #     print('第{}月的活跃人数:{}'.format(i,huoyue_number))
    #     print('第{}月的活跃率:{}'.format(i,huoyue_ratio))

    # 统计表 index:
    list_temp = [date.strftime('%Y/%m'), ac_number, huoyue_number, ac_sum, huoyue_ratio]
    df_list.append(list_temp)
df_activity = pd.DataFrame(df_list, columns=['日期(月)', '参与学生人数', '活跃人数', '学习活动总数', '学生活跃率'])
df_activity



df = pd.read_csv('dxwl.csv')
# print(df.head())
# print(df.shape)
df.sort_values(by='时间')
df_dropTeacher = df.drop(df[(df['账号'] < 20000000000)].index)
start_time = pd.to_datetime(df['时间'].iloc[0].split(" ")[0])
end_time = pd.to_datetime(df['时间'].iloc[df['时间'].shape[0] - 1].split(" ")[0])
# df的时间格式转换
df_dropTeacher['时间'] = pd.to_datetime(df_dropTeacher['时间'])
# 构造时间段序列
date_range2 = pd.date_range(start=start_time, end=end_time, freq="M")
date_range2

# 区分不同的行为：学习行为和缺席
dict_activit = {'参与抢答': 0, '获得考试成绩': 1, '参与签到': 2, '学生访问章节': 3, '完成任务点': 4,
                '批阅互评作业': 5, '提交作业': 6, '提交考试': 7,
                '讨论': 8, '选人被选中': 9, '获得作业成绩': 10, '病假': 11, '签到已过期': 12, '缺勤': 13, '公假': 14,
                '事假': 15}
activit_list = list(dict_activit.keys())

activit_list.insert(0, '日期')
activit_list.insert(0, '姓名')

header = activit_list
print(header)

new_list = []
times = 0
# 每个人的统计
for stu, info1 in df_dropTeacher.groupby('姓名'):

    # 每一周统计
    for i in range(len(date_range2)):
        if (i == 0):
            df_M = info1[info1['时间'] < date_range2[i]]
        else:
            df_M = info1[(info1['时间'] < date_range2[i]) & (info1['时间'] > date_range2[i - 1])]
        #         print(df_W)

        stu_list = [0] * 18
        stu_list[0] = stu.strip('\t')
        stu_list[1] = date_range2[i]

        for course, info2 in df_M.groupby('事件类型'):
            index = dict_activit[course.strip('\t')]
            count_num = info2['时间'].count()
            stu_list[index + 2] = count_num
        #             print(f'{index} {count_num} {course}\n{info2}\n')

        new_list.append(stu_list)
#         print(f'{stu}\n{date_range2[i]}\n')

#     times = times+1
#     if(times == 10):
#         break
new_list
df_month = pd.DataFrame(new_list, columns=header)
abscense = df_month.iloc[:, 13:18]
new_df_month = df_month.iloc[:, :13]
ab_sum = abscense.sum(axis=1)
# print(ab_sum)
new_df_month.insert(13, '缺席', ab_sum)
new_df_week








import pandas as pd
from sqlalchemy import create_engine

# 假设你已经有一个名为 'df_activity' 的 DataFrame

# MySQL数据库的连接信息
host = '10.170.14.187'
user = 'root'
password = 'root'
database = 'dxwl_data'  # 你的数据库名称

# 创建数据库连接
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')

# 将 DataFrame 写入 MySQL 数据库表
df_activity.to_sql('activity_month', con=engine, if_exists='replace', index=False)

