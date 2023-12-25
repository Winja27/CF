import pandas as pd
import matplotlib.pyplot as plt

# 定义归一化函数
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_data():
    score_1 = pd.read_csv('students_score_1.csv', encoding='GBK')
    score_2 = pd.read_csv('students_score_2.csv', encoding='GBK')
    score_1_melted = pd.melt(score_1, id_vars=['姓名', '性别', '学号'],
                             value_vars=['文学', '政治', '军事学', '管理学', '工程学', '哲学', '体育', '高等数学',
                                         '礼仪课'])
    score_2_melted = pd.melt(score_2, id_vars=['姓名', '性别', '学号'],
                             value_vars=['大数据可视化', '计算机组成原理', '推荐系统', 'Python编程', '人工智能',
                                         '机器学习导论', '金融投资课程', '演讲与口才课程'])
    result = pd.concat([score_1_melted, score_2_melted], axis=0)
    result.reset_index(drop=True, inplace=True)
    result['value'] = result.groupby('variable')['value'].transform(normalize)
    result.to_csv('result_data.csv', index=False)
    return result


def best3(df):
    # 删除缺考和挂科的行
    df = df.dropna()
    df = df[df['value'] >= 0.6]
    df_mean = df.groupby('姓名')['value'].mean()
    df_top3 = df_mean.sort_values(ascending=False).head(3)
    print(df_top3)

def boxplot(df):
    df.boxplot(column='value', by='variable')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

def hist(df):
    df['value'].hist(bins=10)
    plt.title('成绩分布')
    plt.xlabel('成绩')
    plt.ylabel('频数')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()
#成绩最高的同学当课代表,如果一个科目的最高成绩有多个同学获得，选择平均成绩最高的同学
def top1(df):
    average_scores = df.groupby('姓名')['value'].mean()

    top_students = df.loc[df.groupby('variable')['value'].idxmax()]

    for subject in top_students['variable'].unique():
        students = top_students[top_students['variable'] == subject]
        if len(students) > 1:
            top_student = students.loc[students['姓名'].map(average_scores).idxmax()]
            top_students = top_students[top_students['variable'] != subject]
            top_students = top_students.append(top_student)

    print(top_students)

#用平均分来衡量这个标准，那么平均分最高的课程可以被认为是最容易的，反之，平均分最低的课程可以被认为是最难的
def easy_difficult(df):
    average_scores = df.groupby('variable')['value'].mean()

    easiest_courses = average_scores.nlargest(3)
    print("最容易的3门课是：")
    print(easiest_courses)

    hardest_courses = average_scores.nsmallest(3)
    print("最难的3门课是：")
    print(hardest_courses)

    average_scores.sort_values().plot(kind='bar')
    plt.ylabel('平均分')
    plt.title('每门课的平均分')
    plt.show()