our
    # df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    # ========================  模型加载  ===========================
    # 请确认需要用到的列名，e.g.:columns = ['cpc','cpm']
    # 例如
    # scaler = joblib.load('./results/scaler.pkl')
    # pca = joblib.load('./results/pca.pkl')
    # data = scaler.transform(data)

    df['timestamp'] = pd.to_datetime(df['timestamp']) 
    df['hours'] = df['timestamp'].dt.hour 
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int) 
    df['cpc X cpm'] = df['cpm'] * df['cpc']
    columns = ['cpc', 'cpm', 'hours','daylight','cpc X cpm'] 
    data = df[columns] 
    scaler = StandardScaler() 
    data = scaler.fit_transform(data) 
    data = pd.DataFrame(data, columns=columns) 
    n_components = 3 
    pca = PCA(n_components=n_components) 
    data = pca.fit_transform(data) 
    data = pd.DataFrame(data,columns=['Dimension' + str(i + 1) for i in range(n_components)])
return data
====================================================================
def get_distance(data, kmeans, n_features):
    """
    计算样本点与聚类中心的距离
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return:每个点距离自己簇中心的距离，Series 类型
    """
    # ====================计算样本点与聚类中心的距离======================== #
    distance = []
    for i in range(0,len(data)):
        point = np.array(data.iloc[i,:n_features])
        center = kmeans.cluster_centers_[kmeans.labels_[i],:n_features]
        distance.append(np.linalg.norm(point - center))
    distance = pd.Series(distance)
return distance
====================================================================
def get_anomaly(data, kmeans, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmean: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
    # ====================检验出样本中的异常点======================= #
    num_anomaly = int(len(data) * ratio)
    new_data = deepcopy(data)
    new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
    threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
    normal = new_data[new_data['is_anomaly'] == 0]
    anormal = new_data[new_data['is_anomaly'] == 1]
return anormal
====================================================================
def predict(preprocess_data):
    """
    该函数将被用于测试，请不要修改函数的输入输出，并按照自己的模型返回相关的数据。
    在函数内部加载 kmeans 模型并使用 get_anomaly 得到每个样本点异常值的判断
    :param preprocess_data: preprocess_data函数的返回值，一般是 DataFrame 类型
    :return:is_anomaly:get_anomaly函数的返回值，各个属性应该为（Dimesion1,Dimension2,......数量取决于具体的pca），distance,is_anomaly，请确保这些列存在
            preprocess_data:  即直接返回输入的数据
            kmeans: 通过joblib加载的对象
            ratio:  异常点的比例，ratio <= 0.03   返回非异常点得分将受到惩罚！
    """
    # 异常值所占比率
    ratio = 0.03
    #KMeans=joblib.load('./results/model.pkl')
    kmeans = KMeans(n_clusters=3, n_init=10, max_iter=800) 
    kmeans.fit(preprocess_data) 
    # 获取异常点数据信息
    is_anomaly = get_anomaly(preprocess_data, kmeans, ratio) 
    return is_anomaly, preprocess_data, kmeans, ratio