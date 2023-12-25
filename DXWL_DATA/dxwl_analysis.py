import pandas as pd


"""*****************每月的总活动统计********************"""
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

"""导出JSON"""
# from json import loads, dumps
# df_activity.to_json('ac_month.json',orient="split",force_ascii=False)
# parsed = loads(res_json)
# dumps(parsed, indent=4)















"""************************班级的每月统计数据*************************"""
df = pd.read_csv('dxwl.csv')
# print(df.head())
# print(df.shape)
df.sort_values(by='时间')
df_dropTeacher = df.drop(df[(df['账号'] < 20000000000)].index)

# 构造数据的header
for x, y in df_dropTeacher.groupby(by=['姓名']):
    list_temp = []
    list_temp.append('时间')
    list_temp.append('姓名')
    list_temp.append('账号')
    for z, s in y.groupby(by=['事件类型']):
        list_temp.append(z.strip('\t'))
    break

df_students_columns = pd.DataFrame([], columns=list_temp)
# print(df_students_columns)
# df_group1 = df_dropTeacher.set_index('时间').groupby('时间')
# for t,info in df_group1:
#     print(t)


# 获取时间开始和结尾日期
start_time = pd.to_datetime(df['时间'].iloc[0].split(" ")[0])
end_time = pd.to_datetime(df['时间'].iloc[df['时间'].shape[0] - 1].split(" ")[0])
print(start_time)
print(end_time)

# 每个月的时间序列
date_range1 = pd.date_range(start=tart_time, end=end_time, freq="M")
print(date_range1)

# df的时间格式转换
df_dropTeacher['时间'] = pd.to_datetime(df_dropTeacher['时间'])

list_new = []
# 根据班级分组
for name1, info in df_dropTeacher.groupby('班级名'):

    # 班级每个月的活动次数，根据时间分组
    for i in range(len(date_range1)):
        if (i == 0):
            df_M = info[info['时间'] < date_range1[i]]
        else:
            df_M = info[(info['时间'] < date_range1[i]) & (info['时间'] > date_range1[i - 1])]
        #         print(df_M)

        list_cl = []
        list_cl.append(name1)
        list_cl.append(date_range1[i].strftime('%Y/%m'))
        jiaohu_person = df_M['姓名'].nunique()
        jiaohu_count = df_M.shape[0]
        ave_jiaohu = (int)(jiaohu_count / jiaohu_person)
        huoyue_person = 0;
        for name2, info1 in df_M.groupby('姓名'):
            if (info1['事件类型'].count() > ave_jiaohu):
                huoyue_person = huoyue_person + 1
        # 活跃人数
        # 活跃率
        list_cl.append(jiaohu_person)
        list_cl.append(format(huoyue_person / jiaohu_person, '.2f'))
        list_cl.append(df_M.shape[0])
        # print(list_cl)
        list_new.append(list_cl)
df_new = pd.DataFrame(list_new, columns=['班级', '日期', '活跃人数', '活跃率', '总活动次数'])
print(df_new)





"""*************************学生每个月的行为统计序列（合并部分维度）*********************"""
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
# new_df_week
new_df_month.to_csv('stu_behavior')


"""输出Json"""
from json import loads, dumps
new_df_month.to_json('stu_month.json',orient="split",force_ascii=False)
# parsed = loads(res_json)
# dumps(parsed, indent=4)









"""**********************每个同学的每周行为数据********************"""
import pandas as pd

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
date_range2 = pd.date_range(start=start_time, end=end_time, freq="W")
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
            df_W = info1[info1['时间'] < date_range2[i]]
        else:
            df_W = info1[(info1['时间'] < date_range2[i]) & (info1['时间'] > date_range2[i - 1])]
        #         print(df_W)

        stu_list = [0] * 18
        stu_list[0] = stu.strip('\t')
        stu_list[1] = date_range2[i]

        for course, info2 in df_W.groupby('事件类型'):
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
df_week = pd.DataFrame(new_list, columns=header)
# df_week.set_index('日期')
# list(df_week.groupby('日期'))
abscense = df_week.iloc[:, 13:18]
new_df_week = df_week.iloc[:, :13]
ab_sum = abscense.sum(axis=1)
print(ab_sum)
new_df_week.insert(13, '缺席', ab_sum)
new_df_week.to_csv('stu_behavior_week')

"""导出JSON"""
from json import loads, dumps
new_df_week.to_json('stu_week.json',orient="split",force_ascii=False)
# parsed = loads(res_json)
# dumps(parsed, indent=4)