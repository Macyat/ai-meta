import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import copy


def PC_Cross_Validation(X, y, pc, days):
    """
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    """
    RMSECV = []
    unique_day = np.unique(days)
    for i in range(pc):
        RMSE = []
        for j in range(len(unique_day)):
            train_index = [k for k in range(len(days)) if days[k] != unique_day[j]]
            test_index = [k for k in range(len(days)) if days[k] == unique_day[j]]

            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if y_train.shape[0] <= i:
                i = y_train.shape[0] - 1

            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex


def Cross_Validation(X, y, pc, days):
    """
    x :光谱矩阵 nxm
    y :浓度阵 （化学值）
    pc:最大主成分数
    cv:交叉验证数量
    return :
           RMSECV:各主成分数对应的RMSECV
    """
    RMSE = []
    unique_day = np.unique(days)
    for j in range(len(unique_day)):
        train_index = [k for k in range(len(days)) if days[k] != unique_day[j]]
        test_index = [k for k in range(len(days)) if days[k] == unique_day[j]]
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean



def CARS_Cloud(X, y, days, end_wave, location, label, model_type, N=30, f=50):
    p = 0.8
    m, n = X.shape
    u = np.power((n / 2), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 2)
    cal_num = np.round(m * p)
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    WaveNum = []
    RMSECV = []
    RIDX = []
    r = []
    for i in range(1, N + 1):
        r.append(u * np.exp(-1 * k * i))
        wave_num = int(np.round(r[i - 1] * n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice(np.arange(m), size=int(cal_num), replace=False)

        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1, -1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData = np.vstack((WaveData, d))
        if wave_num < f:
            f = wave_num
        if ycal.shape[0] < f:
            f = ycal.shape[0]

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)

        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=1).flatten()

        _, rindex = PC_Cross_Validation(xcal, ycal, f, days[cal_index])
        RMSECV.append(Cross_Validation(xcal, ycal, rindex + 1, days[cal_index]))
        RIDX.append(rindex)

    WAVE = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        WD = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
            else:
                WD[j] = wd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))

    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    fig = plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(211)
    #     plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel("被选择的波长数量", fontsize=fonts)
    plt.title("最佳迭代次数：" + str(MinIndex) + "次", fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)

    plt.subplot(212)
    plt.xlabel("蒙特卡洛迭代次数", fontsize=fonts)
    plt.ylabel("RMSECV", fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)

    plt.savefig("cars\\" + location + "_" + label + "_" + model_type + "_cars.png")

    fig2 = plt.figure()
    plt.plot(list(range(end_wave + 1 - len(X[0]), end_wave + 1)), X[40, :])
    plt.scatter(
        [end_wave + 1 - len(X[0]) + i for i in OptWave],
        X[40, OptWave],
        marker="s",
        color="r",
    )
    plt.legend(["First object", "Selected variables"])
    plt.xlabel("Variable index")
    plt.grid(True)
    plt.savefig("cars\\" + location + "_" + label + "_" + model_type + "_selected.png")

    return OptWave, np.mean(RMSECV), RIDX[np.argmin(RMSECV)]
