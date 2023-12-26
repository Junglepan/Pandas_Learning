import pandas as pd

df = pd.read_csv('dxwl.csv')
# print(df.head())
# print(df.shape)
df.sort_values(by='时间')
df_dropTeacher = df.drop(df[(df['账号'] < 20000000000)].index)

# 构造数据的header
for x, y in df_dropTeacher.groupby(by='姓名'):
    list_temp = []
    list_temp.append('时间')
    list_temp.append('姓名')
    list_temp.append('账号')
    for z, s in y.groupby(by='事件类型'):
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
date_range1 = pd.date_range(start=start_time, end=end_time, freq="M")
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

from sqlalchemy import create_engine,types

# 假设你已经有一个名为 'df_activity' 的 DataFrame

# MySQL数据库的连接信息
host = '10.170.14.187'
user = 'root'
password = 'root'
database = 'dxwl_data'  # 你的数据库名称

# 创建数据库连接
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}?charset=utf8mb4')

# 指定数据字段类型
dtype_mapping = {'班级': types.VARCHAR(255, collation='utf8mb4_unicode_ci')}

# 将 DataFrame 写入 MySQL 数据库表
df_new.to_sql('class_month', con=engine, if_exists='replace', index=False,dtype=dtype_mapping)