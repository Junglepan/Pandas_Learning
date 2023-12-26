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

df_dropTeacher.head()
df_dropTeacher.info()


header = ['学院','参与人数','人数占比']
partment_list = []
for name,info in df_dropTeacher.groupby('group1'):
    list_temp = []
    list_temp.append(name.strip("\t"))
    list_temp.append(info['账号'].nunique())
    list_temp.append(format(info['账号'].nunique()/df_dropTeacher['账号'].nunique(),'.3f'))
    partment_list.append(list_temp)
partment_count = pd.DataFrame(partment_list,columns = header)
partment_count.sort_values('参与人数',ascending=False)



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
dtype_mapping = {'学院': types.VARCHAR(255, collation='utf8mb4_unicode_ci')}

# 将 DataFrame 写入 MySQL 数据库表
partment_count.to_sql('partment_count', con=engine, if_exists='replace', index=False,dtype=dtype_mapping)