# -*- coding:utf8 -*-
import os
import csv
import pandas as pd

path_train = "data/dm/train.csv"  # 训练文件
path_test = "data/dm/test.csv"  # 测试文件
my_result = "model/test.csv"
path_train_feature = "train_feature.csv"

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


train_total_num = 68000 # 23734760
test_total_num = 68000
chunk_step = 10000
least_trip_segment = 20  # 最低需要20分钟以上的数据，才能有效分析


def dip_data_per_trip(tripdata):
    ''' 用本函数来判断该段行程司机的危险系数 '''
    import numpy as np
    df_trip = tripdata.sort_values('TIME')
    max_speed = df_trip['SPEED'].max()
    mean_speed = df_trip['SPEED'].mean()
    thre_speed = min(mean_speed*1.2, (max_speed+mean_speed)/2)
    dangerous_speed = max_speed
    size_trip = tripdata['SPEED'].size
    ratio_speed = np.sum(df_trip['SPEED'] > thre_speed)/size_trip
    sum_direction = 0
    df_direciont = df_trip['DIRECTION'].data.tolist()
    for i in range(0, size_trip-1):
        sd = np.abs(df_direciont[i+1] - df_direciont[i])
        # slong = np.abs(df_trip['LONGITUDE'][i + 1] - df_trip['LONGITUDE'][i])
        # slat = np.abs(df_trip['LATITUDE'][i + 1] - df_trip['LATITUDE'][i])
        if(sd > 180):
            sd = 360-sd
        sum_direction = sum_direction+sd
    entropy_direction = sum_direction/(size_trip*180)
    # consistent_location =
    mean_long = df_trip['LONGITUDE'].mean()
    mean_lat = df_trip['LATITUDE'].mean()
    return max_speed, ratio_speed, entropy_direction, mean_long, mean_lat


def dip_data_per_user(userdata):
    '''  先判断Y是否赔付率大于0 '''
    import numpy as np
    user_trip_size = userdata.groupby('TRIP_ID').size()
    meansize = np.mean(user_trip_size)
    thresize = max(least_trip_segment, meansize)
    # init_index = userdata['Y'].index[0]
    cumsize = np.cumsum(user_trip_size).data.tolist()
    length = user_trip_size.size
    user_trip_size = user_trip_size.data.tolist()

    max_speed = 0
    ratio_speed = 0
    entropy_direction = 0
    mean_long = 0
    mean_lat = 0
    threshold = 0
    for i in range(-1, length-1):
        if (user_trip_size[i + 1] >= thresize):
            if(i == -1):
                max_speed_curr, ratio_speed_curr, entropy_direction_curr, mean_long_curr, mean_lat_curr \
                    = dip_data_per_trip(userdata[0:cumsize[0]])
            else:
                max_speed_curr, ratio_speed_curr, entropy_direction_curr, mean_long_curr, mean_lat_curr \
                    = dip_data_per_trip(userdata[cumsize[i]:cumsize[i + 1]])

            evaluate_curr = max_speed_curr + max_speed_curr * ratio_speed_curr + max_speed_curr * entropy_direction_curr
            if (evaluate_curr > threshold):
                max_speed = max_speed_curr
                ratio_speed = ratio_speed_curr
                entropy_direction = entropy_direction_curr
                mean_long = mean_long_curr
                mean_lat = mean_lat_curr
                threshold = evaluate_curr

    return max_speed, ratio_speed, entropy_direction, mean_long, mean_lat


def dip_data(chunk1):
    import numpy as np
    chunksize = chunk1.groupby('TERMINALNO').size()  # 这里可以手写优化
    cumindex = np.cumsum(chunksize).data.tolist()
    userNum = chunksize.size

    rows_list = []
    for i in range(-1, userNum-1):
        RecordtoAdd = {}
        if(i==-1):
            userdf = chunk1[0:cumindex[0]]
        else:
            userdf = chunk1[cumindex[i]:cumindex[i+1]]

        max_speed, ratio_speed, entropy_direction, mean_long, mean_lat = dip_data_per_user(userdf)
        RecordtoAdd.update({'TERMINALNO': userdf['TERMINALNO'][userdf['TERMINALNO'].index[0]]})
        RecordtoAdd.update({'LONGITUDE': mean_long})
        RecordtoAdd.update({'LATITUDE': mean_lat})
        RecordtoAdd.update({'DIRECTION': entropy_direction})
        RecordtoAdd.update({'HEIGHT': ratio_speed})
        RecordtoAdd.update({'SPEED': max_speed})
        RecordtoAdd.update({'Y': userdf['Y'][userdf['Y'].index[0]]})

        rows_list.append(RecordtoAdd)

    return rows_list


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    dtypes = {
        'TERMINALNO': 'uint32',
        'TIME': 'uint32',
        'TRIP_ID': 'uint8',
        'LONGITUDE': 'float32',
        'LATITUDE': 'float32',
        'DIRECTION': 'uint16',
        'HEIGHT': 'float32',
        'SPEED': 'float32',
        'CALLSTATE': 'uint8',
        'Y': 'float32'
    }
    df = pd.read_csv(path_train, iterator=True, dtype=dtypes)
    # df.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
    #                   "CALLSTATE", "Y"]
    rows_list = []
    for i in range(0, train_total_num, chunk_step):
        chunk = df.get_chunk(chunk_step)
        dict_data = dip_data(chunk)
        if(rows_list == []):
            rows_list = dict_data
        else:
            dict1 = rows_list[-1]
            dict2 = dict_data[0]
            if(dict1['TERMINALNO'] != dict2['TERMINALNO']):
                rows_list = rows_list + dict_data
            elif(dict1['Y']==0):
                rows_list.pop()
                rows_list = rows_list + dict_data
            else:
                if(dict1['SPEED'] > dict2['SPEED']):
                    dict_data[0] = dict1
                rows_list.pop()
                rows_list = rows_list + dict_data
    df = pd.DataFrame(rows_list)
    df.to_csv(path_train_feature, index=False)


def train():
    dtypes = {
        'TERMINALNO': 'uint32',
        'LONGITUDE': 'float32',
        'LATITUDE': 'float32',
        'DIRECTION': 'float32',
        'HEIGHT': 'float32',
        'SPEED': 'float32',
        'Y': 'float32'
    }
    df = pd.read_csv(path_train_feature, dtype=dtypes)

    dfsize = df['SPEED'].size
    positiveNum = int(min((int(df.groupby('Y').size().size)-1)/2, 300))
    df = df.sort_values('SPEED', 0, False)
    # df_index = df['SPEED'].index
    max_speed = df['SPEED'].max()

    import numpy as np
    possibility = np.zeros((6, 2))
    X1 = np.zeros((positiveNum, 3))
    X2 = np.zeros((positiveNum, 3))
    Y = np.zeros((positiveNum, 1))
    count_positive = 0
    count_negative = 0
    last_positive_num = 0
    for i in range(1, dfsize):
        if((count_positive < positiveNum) & (df['Y'][df['Y'].index[i-1]]>0.001)):
            X1[count_positive][0] = df['SPEED'][df['Y'].index[i-1]]/max_speed
            X1[count_positive][1] = df['DIRECTION'][df['Y'].index[i-1]]
            X1[count_positive][2] = df['HEIGHT'][df['Y'].index[i-1]]
            Y[count_positive][0] = df['Y'][df['Y'].index[i-1]]
            count_positive = count_positive+1
            if (np.mod(count_positive, np.ceil(positiveNum / 5)) == 0):
                index = int(count_positive / np.ceil(positiveNum / 5))
                possibility[index - 1][0] = X1[count_positive-1][0]
                if(index == 1):
                    possibility[index - 1][1] = count_positive/(count_positive+count_negative)
                    last_positive_num = count_positive
                else:
                    a1 = count_positive - last_positive_num
                    a2 = count_positive+count_negative-last_positive_num/possibility[index - 2][1]
                    possibility[index - 1][1] = a1/(a1+a2)

        elif((df['Y'][df['Y'].index[i-1]]<0.001)):
            mod_negative = np.mod(count_negative, positiveNum)
            X2[mod_negative][0] = df['SPEED'][df['Y'].index[i-1]]/max_speed
            X2[mod_negative][1] = df['DIRECTION'][df['Y'].index[i-1]]
            X2[mod_negative][2] = df['HEIGHT'][df['Y'].index[i-1]]
            count_negative = count_negative +1


        if(count_positive == positiveNum & count_negative >= positiveNum):
            possibility[5][0] = 0
            possibility[5][1] = count_positive/(dfsize-count_positive-count_negative)
            break

##################################################################################
    X = np.concatenate((X1, X2), axis=0)
    Ylabel = np.concatenate((np.ones((X1.shape[0], 1)), np.zeros((X2.shape[0], 1))), axis=0)

    from sklearn.ensemble import RandomForestClassifier
    # param = [{'solver': 'adam', 'learning_rate_init': 0.01}]
    # max_iter = 1000
    classifier =RandomForestClassifier(max_depth=3, random_state=0)
    classifier.fit(X, Ylabel)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=3, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                           oob_score=False, random_state=0, verbose=True, warm_start=False)
##################################################################################
    from sklearn.neural_network import MLPRegressor
    regression = MLPRegressor(
        hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    regression.fit(X1, Y)

##################################################################################
    from sklearn.externals import joblib
    filename = 'classifier_model.sav'
    joblib.dump(classifier, filename)
    filename = 'regression_model.sav'
    joblib.dump(regression, filename)
    np.save('possibility.npy', possibility)


def test():
    from extract_test_features import read_csv_test
    read_csv_test()
    out_test = "my_test.csv"  # 测试文件
    dtypes = {
        'TERMINALNO': 'uint32',
        'LONGITUDE': 'float32',
        'LATITUDE': 'float32',
        'DIRECTION': 'float32',
        'HEIGHT': 'float32',
        'SPEED': 'float32',
    }
    df = pd.read_csv(out_test, dtype=dtypes)
    dfsize = df['SPEED'].size

    import numpy as np
    X1 = np.zeros((dfsize, 3))
    max_speed = df['SPEED'].max()
    for i in range(0, dfsize):
        X1[i][0] = df['SPEED'][i] / max_speed
        X1[i][1] = df['DIRECTION'][i]
        X1[i][2] = df['HEIGHT'][i]

    possibility = np.load('possibility.npy')
    from sklearn.neural_network import MLPClassifier
    from sklearn.neural_network import MLPRegressor
    from sklearn.externals import joblib
    filename = 'classifier_model.sav'
    classifier = joblib.load(filename)
    filename = 'regression_model.sav'
    regression = joblib.load(filename)
    y1 = classifier.predict_proba(X1)
    # np.save('y1.npy', y1);
    rows_list = []
    for i in range(1, dfsize+1):
        print(y1[i-1][1])
        speed_curr = X1[i - 1][0]
        RecordtoAdd = {}
        if(speed_curr>possibility[0][0]):
            if(np.random.rand()<possibility[0][1]):
                regression_data = regression.predict(np.array([X1[i-1]]))
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i-1]], 'Pred' : regression_data.tolist().pop()})
            else:
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': 0})
        elif(speed_curr>possibility[1][0]):
            if(np.random.rand()<possibility[1][1]):
                regression_data = regression.predict(np.array([X1[i - 1]]))
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': regression_data.tolist().pop()})
            else:
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': 0})
        elif(speed_curr>possibility[2][0]):
            if(np.random.rand()<possibility[2][1]):
                regression_data = regression.predict(np.array([X1[i-1]]))
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': regression_data.tolist().pop()})
            else:
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': 0})
        elif(speed_curr>possibility[3][0]):
            if(np.random.rand()<possibility[3][1]):
                regression_data = regression.predict(np.array([X1[i-1]]))
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': regression_data.tolist().pop()})
            else:
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': 0})
        elif(speed_curr>possibility[4][0]):
            if(np.random.rand()<possibility[4][1]):
                regression_data = regression.predict(np.array([X1[i-1]]))
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': regression_data.tolist().pop()})
            else:
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': 0})
        else:
            if(np.random.rand()<possibility[5][1]):
                regression_data = regression.predict(np.array([X1[i-1]]))
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': regression_data.tolist().pop()})
            else:
                RecordtoAdd.update({'Id': df['TERMINALNO'][df['TERMINALNO'].index[i - 1]], 'Pred': 0})
        rows_list.append(RecordtoAdd)
    df = pd.DataFrame(rows_list)
    df.to_csv(my_result, index=False)


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np
    count = 0
    with open(path_train_feature) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], my_result[count]])  # 随机值
                count = count+1

                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    # process()
    read_csv()
    train()
    test()
   # process()
