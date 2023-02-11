import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from typing import List
import ctypes
from ctypes import *
import pathlib
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

script_dir = os.path.abspath(os.path.dirname("CppLib\cmake-build-debug\\"))
lib_path = os.path.join(script_dir, "CppLib.dll")

# print(script_dir, lib_path)

# import shared lib
c_lib = cdll.LoadLibrary(lib_path)

# Config import Modele Lineaire
linear_model = c_lib.LinearModelTrain
linear_model.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         c_int, c_int, c_int, c_int]
linear_model.restype = ctypes.POINTER(ctypes.c_float)

# Config import PMC
createPMC = c_lib.createPMC
createPMC.argtypes = [ctypes.POINTER(ctypes.c_int), c_int]
createPMC.restype = ctypes.c_void_p

trainPMC = c_lib.trainPMC
trainPMC.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                     ctypes.POINTER(ctypes.c_float),
                     c_int, c_int, c_int, c_bool]

predictPMC = c_lib.predictPMC
predictPMC.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_bool]
predictPMC.restype = ctypes.POINTER(ctypes.c_float)

freeMemory = c_lib.freeMemory
freeMemory.argtypes = [ctypes.c_void_p]

checkPMC = c_lib.checkPMC
checkPMC.argtypes = [ctypes.c_void_p]

savePMC = c_lib.savePMC
savePMC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

createPMCFromFile = c_lib.createPMCFromFile
createPMCFromFile.argtypes = [ctypes.c_void_p]
createPMCFromFile.restype = ctypes.c_void_p

saveML = c_lib.saveML
saveML.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]

saveRBF = c_lib.saveRBF
saveRBF.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int]

loadML = c_lib.loadML
loadML.argtypes = [ctypes.c_void_p]
loadML.restype = ctypes.POINTER(ctypes.c_float)

loadRBF = c_lib.loadRBF
loadRBF.argtypes = [ctypes.c_void_p]
loadRBF.restype = ctypes.POINTER(ctypes.c_float)

loadInfoML = c_lib.loadInfoML
loadInfoML.argtypes = [ctypes.c_void_p]
loadInfoML.restype = ctypes.POINTER(ctypes.c_int)

loadInfoRBF = c_lib.loadInfoRBF
loadInfoRBF.argtypes = [ctypes.c_void_p]
loadInfoRBF.restype = ctypes.POINTER(ctypes.c_int)

kMean = c_lib.kMeansAlgo
kMean.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
kMean.restype = ctypes.POINTER(ctypes.c_float)


# ML
def TrainModeleLineaire(nb_rep, alpha, is_classification, filename=b"linear_model_save.txt"):
    if not is_classification:
        res = linear_model(nb_rep, alpha, points_c, classes_c, len(X), len(X[0]), len([[c] for c in Y][0]), 0,
                           is_classification)

        W = res[:len(X[0]) + 1]
        W = [1.0, *W]

        Wc = (c_float * len(W))(*W)
        SaveModeleLineaire(filename, Wc, len(X[0]) + 1, len(Y[0]))
        return W

    W = list()
    if (len(Y[0]) < 2):
        res = linear_model(nb_rep, alpha, points_c, classes_c, len(X), len(X[0]), len([[c] for c in Y][0]), 0,
                           is_classification)

        W = res[:len(X[0]) + 1]
        # print("W = ", W)
        Wc = (c_float * len(W))(*W)
    else:
        for i in range(len(Y[0])):
            res = linear_model(nb_rep, alpha, points_c, classes_c, len(X), len(X[0]), len(Y[0]), i,
                               is_classification)

            Wr = res[:len(X[0]) + 1]
            # print("Wr = ", W)
            W.append(Wr)

        Wc = [W[i][j] for i in range(len(W)) for j in range(len(W[i]))]
        Wc = (c_float * len(Wc))(*Wc)

    SaveModeleLineaire(filename, Wc, len(X[0]) + 1, len(Y[0]))
    print("Linear Model Training Done")
    return W


def PredictModeleLineaire(input, W, is_classification, results):
    # Model Lineaire C++
    # print("Prediction")
    if not is_classification:
        return np.matmul(np.transpose(W), np.array([1.0, *input]))

    if isinstance(W[0], list) or len(set(results)) > 2:
        maxMat = [np.matmul(np.transpose(Wt), np.array([1.0, *input])) for Wt in W]
        index = maxMat.index(max(maxMat))

        # print(list(reversed(sorted([(maxMat[i], results[i]) for i in range(len(maxMat))], key=lambda x: x[0],))))
        c = results[index]
        # print(maxMat[results.index("France")])
        # print(maxMat[index])
    else:
        c = 'lightcyan' if np.matmul(np.transpose(W), np.array([1.0, *input])) >= 0 else 'pink'
    return c


def SaveModeleLineaire(filename, W, col, row):
    t = ctypes.create_string_buffer(filename)
    saveML(t, W, col, row)


def LoadModeleLineaire(filename):
    t = ctypes.create_string_buffer(filename)
    size = loadInfoML(t)[0]
    yCol = loadInfoML(t)[1]
    res = loadML(t)
    if yCol > 2:
        W = []
        for i in range(yCol):
            W.append(res[i * size:i * size + size])
    else:
        W = res[:size]
    return W


def showGraphModeleLineaire(W, is_classification, borne, results, is_3d=False):
    print(results)
    if is_classification:
        test_points = []
        test_colors = []
        for row in range(borne[0], borne[1]):
            for col in range(borne[0], borne[1]):
                p = np.array([col / 100, row / 100])
                p_c = (c_float * len(p))(*p)
                c = PredictModeleLineaire(p, W, is_classification, results)
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        if is_3d:
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
            ax.scatter(test_points[:, 0], test_points[:, 1], test_colors[:, 0], c="pink")
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c='blue')
        else:
            plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
            plt.scatter(X[:, 0], X[:, 1], c=colors)

    else:
        test_points_x = []
        test_points_y = []

        if is_3d:
            for line in range(borne[0], borne[1]):
                for row in range(borne[0], borne[1]):
                    p = [line / 100, row / 100]
                    t = PredictModeleLineaire(p, W, is_classification, results)
                    test_points_x.append(p)
                    test_points_y.append(t)
            test_points_x = np.array(test_points_x)
            test_points_y = np.array(test_points_y)
            ax.scatter(test_points_x[:, 0], test_points_x[:, 1], test_points_y, c="pink")
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c='blue')
        else:
            for line in range(borne[0], borne[1]):
                p = np.array([line / 100])
                t = PredictModeleLineaire(p, W, is_classification, results)
                test_points_x.append(p)
                test_points_y.append(t)
            plt.scatter(test_points_x, test_points_y, c="r")
            plt.scatter(X, Y)
    plt.show()


# PMC
def CreatePMC(list_npl):
    npl = list_npl
    npl_c = (c_int * len(npl))(*npl)

    pmc_r = createPMC(npl_c, len(npl))
    # print(pmc_r)

    return pmc_r


def CreatePMCFromFile(file):
    t = ctypes.create_string_buffer(file)
    return createPMCFromFile(t)


def TrainPMC(pmc, nb_rep, alpha, is_classification):
    trainPMC(pmc, nb_rep, alpha, points_c, classes_c, len(X), len(X[0]), len(Y[0]), is_classification)

    return pmc


def SavePMC(pmc, file):
    t = ctypes.create_string_buffer(file)
    savePMC(pmc, t)


def PredictPMC(pmc, input, is_classification, nb_results, results):
    # print(pmc)
    input_c = (c_float * len(input))(*input)
    t = predictPMC(pmc, input_c, is_classification)[1:nb_results + 1]

    if not is_classification:
        return t[0]

    if nb_results > 2:
        maxMat = [t[i] for i in range(nb_results)]
        # print(list(reversed(sorted([(maxMat[i], results[i]) for i in range(len(maxMat))], key=lambda x: x[0], ))))
        index = maxMat.index(max(maxMat))
        c = results[index]
    else:
        c = 'pink' if t[0] <= 0 else 'lightcyan'

    return c


def FreePMC(pmc):
    freeMemory(pmc)


def showGraphPMC(pmc, is_classification, borne, nb_results, results, is_3d=False):
    if is_classification:
        test_points = []
        test_colors = []
        for row in range(borne[0], borne[1]):
            for col in range(borne[0], borne[1]):
                p = np.array([col / 100, row / 100])
                p_c = (c_float * len(p))(*p)
                c = PredictPMC(pmc, p, is_classification, nb_results, results)
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        if is_3d:
            ax.scatter(test_points[:, 0], test_points[:, 1], test_colors[:, 0], c="pink")
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c='blue')
        else:
            plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
            plt.scatter(X[:, 0], X[:, 1], c=colors)

    else:
        test_points_x = []
        test_points_y = []
        if is_3d:
            for line in range(borne[0], borne[1]):
                for row in range(borne[0], borne[1]):
                    p = np.array([line / 100, row / 100])
                    t = PredictPMC(pmc, p, is_classification, nb_results, results)
                    test_points_x.append(p)
                    test_points_y.append(t)
            test_points_x = np.array(test_points_x)
            test_points_y = np.array(test_points_y)
            ax.scatter(test_points_x[:, 0], test_points_x[:, 1], test_points_y, c="pink")
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c='blue')
        else:
            for line in range(borne[0], borne[1]):
                p = np.array([line / 100])
                t = PredictPMC(pmc, p, is_classification, nb_results, results)
                test_points_x.append(p)
                test_points_y.append(t)
            plt.scatter(test_points_x, test_points_y, c="r")
            plt.scatter(X, Y)
    plt.show()


# RBF
def GetKMeans(k):
    uks = []
    kMeanMat = kMean(points_c, k, len(X), len(X[0]))
    for k in range(k):
        uks.append(kMeanMat[k * len(X[0]): k * len(X[0]) + len(X[0])])

    return uks


def TrainRBF(gamma, uks, nb_results, filename=b"rbf_save.txt"):
    sigma = []

    for i in range(len(X)):
        sigma.append([])
        for k in range(nb_results):
            uk = uks[k]
            # print(uk)
            xk = np.array([a - b for a, b in zip(X[i], uk)])
            # xk = np.array([a-b for a, b in zip(X[i], X[k])])
            norm = sum([elem ** 2 for elem in xk])
            gauss = math.exp(-gamma * norm)

            sigma[i].append(gauss)

    sigma = np.array(sigma)

    sigmaT = np.transpose(sigma)

    # print(sigma)

    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(sigmaT, sigma)), sigmaT), Y)
    # W = np.matmul(np.linalg.inv(sigma), Y)

    # print(W)

    Wc = [W[i][j] for i in range(len(W)) for j in range(len(W[i]))]
    Wc = (c_float * len(Wc))(*Wc)
    SaveRBF(filename, Wc, uks, len(W[0]), len(W))
    return W


def SaveRBF(filename, W, uks, col, row):
    t = ctypes.create_string_buffer(filename)

    uks_c = [uks[i][j] for i in range(len(uks)) for j in range(len(uks[i]))]
    uks_c = (c_float * len(uks_c))(*uks_c)

    saveRBF(t, W, uks_c, col, row, len(uks[0]), len(uks))


def LoadRBF(filename):
    t = ctypes.create_string_buffer(filename)
    size = loadInfoRBF(t)[0]
    yCol = loadInfoRBF(t)[1]
    xUKS = loadInfoRBF(t)[2]
    yUKS = loadInfoRBF(t)[3]
    res = loadRBF(t)
    uks = []

    for i in range(yUKS):
        uks.append(res[i * xUKS: i * xUKS + xUKS])

    W = []
    for i in range(yCol):
        W.append(res[(xUKS * yUKS) + i * size: (xUKS * yUKS) + i * size + size])
    return W, uks


def PredictRBF(input, W, is_classification, gamma, uks, nb_results, results):
    # Model Lineaire C++
    # print("Prediction RBF")
    if not is_classification:
        out = 0
        for k in range(len(W[0])):
            for n in range(len(W)):
                uk = uks[n]
                xk = np.array([a - b for a, b in zip(input, uk)])
                norm = sum([elem ** 2 for elem in xk])
                gauss = math.exp(-gamma * norm)
                out += W[n][k] * gauss
        return out

    if nb_results > 2:
        maxMat = []
        for k in range(len(W[0])):
            out = 0
            for n in range(len(W)):
                uk = uks[n]
                xk = np.array([a - b for a, b in zip(input, uk)])
                norm = sum([elem ** 2 for elem in xk])
                gauss = math.exp(-gamma * norm)
                out += W[n][k] * gauss
            maxMat.append(out)

        index = maxMat.index(max(maxMat))

        # print(list(reversed(sorted([(maxMat[i], results[i]) for i in range(len(maxMat))], key=lambda x: x[0],))))
        c = results[index]
        # print(maxMat[results.index("France")])
        # print(maxMat[index])
    else:
        out = 0
        for k in range(len(W[0])):
            out = 0
            for n in range(len(W)):
                uk = uks[n]
                xk = np.array([a - b for a, b in zip(input, uk)])
                norm = sum([elem ** 2 for elem in xk])
                gauss = math.exp(-gamma * norm)
                out += W[n][k] * gauss
        c = 'lightcyan' if out >= 0 else 'pink'
    return c


def showGraphRBF(W, is_classification, borne, gamma, uks, nb_results, results, is_3d=False):
    if is_classification:
        test_points = []
        test_colors = []
        for row in range(borne[0], borne[1]):
            for col in range(borne[0], borne[1]):
                p = np.array([col / 100, row / 100])
                p_c = (c_float * len(p))(*p)
                c = PredictRBF(p, W, is_classification, gamma, uks, nb_results, results)
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        if is_3d:
            ax.scatter(test_points[:, 0], test_points[:, 1], test_colors[:, 0], c="pink")
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c='blue')
        else:
            plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
            plt.scatter(X[:, 0], X[:, 1], c=colors)

    else:
        test_points_x = []
        test_points_y = []
        if is_3d:
            for line in range(borne[0], borne[1]):
                for row in range(borne[0], borne[1]):
                    p = np.array([line / 100, row / 100])
                    t = PredictRBF(p, W, is_classification, gamma, uks, nb_results, results)
                    test_points_x.append(p)
                    test_points_y.append(t)
            test_points_x = np.array(test_points_x)
            test_points_y = np.array(test_points_y)
            ax.scatter(test_points_x[:, 0], test_points_x[:, 1], test_points_y, c="pink")
            ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c='blue')
        else:
            for line in range(borne[0], borne[1]):
                p = np.array([line / 100])
                t = PredictRBF(p, W, is_classification, gamma, uks, nb_results, results)
                test_points_x.append(p)
                test_points_y.append(t)
            plt.scatter(test_points_x, test_points_y, c="r")
            plt.scatter(X, Y)
    plt.show()


# Image
def convertHistogram(image):
    return [elem / (image.size[0] * image.size[1]) * 100 for elem in image.histogram()]


def rgbAverage(image):
    r, g, b = 0, 0, 0
    div = 0

    for color in image.getcolors(image.size[0] * image.size[1]):
        r += color[0] * color[1][0]
        g += color[0] * color[1][1]
        b += color[0] * color[1][2]
        div += color[0]
    r /= div
    g /= div
    b /= div

    return [r, g, b]


def averageRGB100(image):
    l10 = []
    ln = [0] * 10
    for k in range(10):
        start = int(image.size[0] / 10 * k)
        end = int(image.size[0] / 10 * (k + 1)) if int(image.size[0] / 10 * (k + 1)) < image.size[0] else image.size[0]

        box = (start, 0, end, image.size[1])
        l10.append(
            rgbAverage(
                image.crop(box)
            )
        )

    ret = []
    for i in range(len(l10)):
        ret.append(l10[i][0])
        ret.append(l10[i][1])
        ret.append(l10[i][2])

    return ret


if __name__ == "__main__":
    W = list()
    Ws = list()

    # # Linear Simple
    # print("Linear Simple")
    #
    # # Création Dataset
    # X = np.array([
    #     [1, 1],
    #     [2, 3],
    #     [3, 3]
    # ])
    # Y = np.array([
    #     [1],
    #     [-1],
    #     [-1]
    # ])
    #
    # borne = (100, 300)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("Linear Simple")
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # # W = TrainModeleLineaire(1000, 0.01, True, b"CasTestSave/LinearSimple_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/LinearSimple_ML.txt")
    # plt.title("Modèle Linéaire - Linear Simple")
    # showGraphModeleLineaire(W, True, borne, colors)
    #
    # # PMC
    # # pmc = CreatePMC([2, 1])
    # # TrainPMC(pmc, 10000, 0.01, True)
    # # SavePMC(pmc, b"CasTestSave/LinearSimple_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/LinearSimple_PMC.txt")
    # plt.title("PMC - Linear Simple")
    # showGraphPMC(pmc, True, borne, len(Y[0]), colors)
    # FreePMC(pmc)
    #
    # # RBF
    # print("RBF")
    #
    # gamma = 0.01
    #
    # # uks = X
    # #
    # # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/LinearSimple_RBF.txt")
    # filename = b"CasTestSave/LinearSimple_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - Linear Simple")
    # showGraphRBF(W, True, borne, gamma, uks, len(Y[0]), colors)

    # #Regression Linéaire
    #
    # #Linear Simple 2D
    # print("Linear Simple 2D")
    #
    # # Création Dataset
    # X = np.array([
    #     [1],
    #     [2]
    # ])
    # Y = np.array([
    #     [2],
    #     [3]
    # ])
    #
    # borne = (100, 200)
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.scatter(X, Y)
    # plt.show()
    # plt.clf()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # PredictModeleLineaire(10000, 0.01, False)
    #
    # # PMC
    # PredictPMC([1, 1], 10000, 0.01, False)

    # # Linear Multiple
    # print("Linear Multiple")
    #
    # # Création Dataset
    # X = np.concatenate(
    #     [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    # Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    #
    # borne = (100, 300)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("Linear Multiple")
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # # W = TrainModeleLineaire(1000, 0.01, True, b"CasTestSave/LinearMultiple_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/LinearMultiple_ML.txt")
    # plt.title("Modèle Linéaire - Linear Multiple")
    # showGraphModeleLineaire(W, True, borne, colors)
    #
    # # PMC
    # # pmc = CreatePMC([2, 1])
    # # TrainPMC(pmc, 10000, 0.01, True)
    # # SavePMC(pmc, b"CasTestSave/LinearMultiple_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/LinearMultiple_PMC.txt")
    # plt.title("PMC - Linear Multiple")
    # showGraphPMC(pmc, True, borne, len(Y[0]), colors)
    # FreePMC(pmc)
    #
    # # RBF
    # print("RBF")
    #
    # gamma = 0.00001
    #
    # # uks = X
    #
    # # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/LinearMultiple_RBF.txt")
    # filename = b"CasTestSave/LinearMultiple_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - Linear Multiple")
    # showGraphRBF(W, True, borne, gamma, uks, len(Y[0]), colors)

    # XOR
    # print("XOR")

    # Création Dataset
    # X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    # Y = np.array([[1], [1], [-1], [-1]])
    #
    # borne = (0, 100)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]

    # plt.title("XOR")
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()

    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)

    # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, True, b"CasTestSave/XOR_ML.txt")

    # W = LoadModeleLineaire(b"CasTestSave/XOR_ML.txt")
    # plt.title("Modèle Linéaire - XOR")
    # showGraphModeleLineaire(W, True, borne, colors)

    # PMC
    # pmc = CreatePMC([2, 2, 1])
    # TrainPMC(pmc, 10000, 0.01, True)
    # SavePMC(pmc, b"CasTestSave/XOR_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/XOR_PMC.txt")
    # plt.title("PMC - XOR")
    # showGraphPMC(pmc, True, borne, len(Y[0]), colors)
    # FreePMC(pmc)

    # RBF
    # print("RBF")
    #
    # gamma = 0.1
    #
    # uks = X
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/XOR_RBF.txt")
    # filename = b"CasTestSave/XOR_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - XOR")
    # showGraphRBF(W, True, borne, gamma, uks, len(Y[0]), colors)

    # # Cross
    # print("Cross")
    #
    # # Création Dataset
    # X = np.random.random((500, 2)) * 2.0 - 1.0
    # Y = np.array([[1] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1] for p in X])
    #
    # borne = (-100, 100)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("Cross")
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)

    # # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, True, b"CasTestSave/Cross_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/Cross_ML.txt")
    # plt.title("Modèle Linéaire - Cross")
    # showGraphModeleLineaire(W, True, borne, colors)
    #
    # # PMC
    # pmc = CreatePMC([2, 4, 1])
    # TrainPMC(pmc, 100000, 0.01, True)
    # SavePMC(pmc, b"CasTestSave/Cross_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/Cross_PMC.txt")
    # plt.title("PMC - Cross")
    # showGraphPMC(pmc, True, borne, len(Y[0]), colors)
    # FreePMC(pmc)

    # # RBF
    # print("RBF")
    #
    # gamma = 1
    #
    # uks = GetKMeans(15)
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/Cross_RBF.txt")
    # filename = b"CasTestSave/Cross_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - Cross")
    # showGraphRBF(W, True, borne, gamma, uks, len(Y[0]), colors)
    #
    # MultiLinear 3 classes
    # print("MultiLinear 3 classes")
    #
    # # Création Dataset
    # X = np.random.random((500, 2)) * 2.0 - 1.0
    # Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
    #               [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
    #               [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
    #               [0, 0, 0] for p in X])
    #
    # X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    # Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    #
    # borne = (-100, 100)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else ["blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    # colorsR = ["lightcyan", "pink", "lightgreen"]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("MultiLinear 3 classes")
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)

    # # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, True, b"CasTestSave/3Classes_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/3Classes_ML.txt")
    # plt.title("Modèle Linéaire - MultiLinear 3 classes")
    # showGraphModeleLineaire(W, True, borne, colorsR)

    # # PMC
    # pmc = CreatePMC([2, 3])
    # TrainPMC(pmc, 100000, 0.01, True)
    # SavePMC(pmc, b"CasTestSave/3Classes_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/3Classes_PMC.txt")
    # plt.title("PMC - MultiLinear 3 classes")
    # showGraphPMC(pmc, True, borne, len(Y[0]), colorsR)
    # FreePMC(pmc)
    #
    # # RBF
    # print("RBF")
    #
    # gamma = 0.001
    #
    # uks = GetKMeans(3)
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/3Classes_RBF.txt")
    # filename = b"CasTestSave/3Classes_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - MultiLinear 3 classes")
    # showGraphRBF(W, True, borne, gamma, uks, len(Y[0]), colorsR)
    #
    # # Multi Cross
    # print("Multi Cross")
    #
    # # Création Dataset
    # X = np.random.random((1000, 2)) * 2.0 - 1.0
    # Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [-1, 1, -1] if abs(
    #     p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [-1, -1, 1] for p in X])
    #
    # borne = (-100, 100)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    # colorsR = ["lightcyan", "pink", "lightgreen"]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("MultiCross")
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)

    # # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, True, b"CasTestSave/MultiCross_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/MultiCross_ML.txt")
    # plt.title("Modèle Linéaire - MultiCross")
    # showGraphModeleLineaire(W, True, borne, colorsR)

    # PMC
    # pmc = CreatePMC([2, 20, 21, 3])
    # TrainPMC(pmc, 1000000, 0.01, True)
    # SavePMC(pmc, b"CasTestSave/MultiCross_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/MultiCross_PMC.txt")
    # plt.title("PMC - MultiCross")
    # showGraphPMC(pmc, True, borne, len(Y[0]), colorsR)
    # FreePMC(pmc)

    # # RBF
    # print("RBF")
    #
    # gamma = 20
    #
    # uks = X
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/MultiCross_RBF.txt")
    # filename = b"CasTestSave/MultiCross_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - MultiCross")
    # showGraphRBF(W, True, borne, gamma, uks, len(Y[0]), colorsR)

    # REGRESSION LINEAIRE

    # # Linear Simple 2D
    # print("Linear Simple 2D")
    #
    # # Création Dataset
    # X = np.array([
    #     [1],
    #     [2]
    # ])
    # Y = np.array([
    #     [2],
    #     [3]
    # ])
    #
    # borne = (0, 300)
    #
    # colors = ["blue"]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("Linear Simple 2D")
    # plt.scatter(X, Y, c=colors)
    # plt.show()
    # plt.clf()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    # #
    # # # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, False, b"CasTestSave/LinearSimple2D_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/LinearSimple2D_ML.txt")
    # plt.title("Modèle Linéaire - Linear Simple 2D")
    # showGraphModeleLineaire(W, False, borne, colors)

    # # PMC
    # pmc = CreatePMC([1,1])
    # TrainPMC(pmc, 10000, 0.01, False)
    # SavePMC(pmc, b"CasTestSave/LinearSimple2D_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/LinearSimple2D_PMC.txt")
    # plt.title("PMC - Linear Simple 2D")
    # showGraphPMC(pmc, False, borne, len(Y[0]), colors)
    # FreePMC(pmc)

    # # RBF
    # print("RBF")
    #
    # gamma = 0.01
    #
    # uks = X
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/LinearSimple2D_RBF.txt")
    # filename = b"CasTestSave/LinearSimple2D_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - Linear Simple 2D")
    # showGraphRBF(W, False, borne, gamma, uks, len(Y[0]), colors)

    # # Non Linear Simple 2D
    # print("Non Linear Simple 2D")
    #
    # # Création Dataset
    # X = np.array([
    #     [1],
    #     [2],
    #     [3]
    # ])
    # Y = np.array([
    #     [2],
    #     [3],
    #     [2.5]
    # ])
    #
    # borne = (0, 300)
    #
    # colors = ["blue"]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.title("Non Linear Simple 2D")
    # plt.scatter(X, Y, c=colors)
    # plt.show()
    # plt.clf()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)

    # # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, False, b"CasTestSave/NonLinearSimple2D_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/NonLinearSimple2D_ML.txt")
    # plt.title("Modèle Linéaire - Non Linear Simple 2D")
    # showGraphModeleLineaire(W, False, borne, colors)

    # # PMC
    # pmc = CreatePMC([1, 15, 1])
    # TrainPMC(pmc, 10000, 0.01, False)
    # SavePMC(pmc, b"CasTestSave/NonLinearSimple2D_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/NonLinearSimple2D_PMC.txt")
    # plt.title("PMC - Non Linear Simple 2D")
    # showGraphPMC(pmc, False, borne, len(Y[0]), colors)
    # FreePMC(pmc)
    #
    # # # RBF
    # print("RBF")
    #
    # gamma = 0.01
    #
    # uks = X
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/NonLinearSimple2D_RBF.txt")
    # filename = b"CasTestSave/NonLinearSimple2D_RBF.txt"
    # W, uks = LoadRBF(filename)
    # plt.title("RBF - Non Linear Simple 2D")
    # showGraphRBF(W, False, borne, gamma, uks, len(Y[0]), colors)

    # Linear Simple 3D
    # print("Linear Simple 3D")
    #
    # # Création Dataset
    # X = np.array([
    #     [1, 1],
    #     [2, 2],
    #     [3, 1]
    # ])
    # Y = np.array([
    #     [2],
    #     [3],
    #     [2.5]
    # ])
    #
    # borne = (0, 300)
    #
    # colors = ["blue"]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("Linear Simple 3D")
    # ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    # plt.show()
    # plt.clf()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # # W = TrainModeleLineaire(1000, 0.01, False, b"CasTestSave/LinearSimple3D_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/LinearSimple3D_ML.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("Modèle Linéaire - Linear Simple 3D")
    # showGraphModeleLineaire(W, False, borne, colors, True)
    #
    # # PMC
    # # pmc = CreatePMC([2, 1])
    # # TrainPMC(pmc, 10000, 0.01, False)
    # # SavePMC(pmc, b"CasTestSave/LinearSimple3D_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/LinearSimple3D_PMC.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("PMC - Linear Simple 3D")
    # showGraphPMC(pmc, False, borne, len(Y[0]), colors, True)
    # FreePMC(pmc)
    #
    # # # RBF
    # print("RBF")
    #
    # gamma = 0.01
    #
    # # uks = X
    # #
    # # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/LinearSimple3D_RBF.txt")
    # filename = b"CasTestSave/LinearSimple3D_RBF.txt"
    # W, uks = LoadRBF(filename)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("RBF - Linear Simple 3D")
    # showGraphRBF(W, False, borne, gamma, uks, len(Y[0]), colors, True)

    # Linear Tricky 3D
    # print("Linear Tricky 3D")
    #
    # # Création Dataset
    # X = np.array([
    #     [1, 1],
    #     [2, 2],
    #     [3, 3]
    # ])
    # Y = np.array([
    #     [1],
    #     [2],
    #     [3]
    # ])
    #
    # borne = (0, 300)
    #
    # colors = ["blue"]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("Linear Tricky 3D")
    # ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    # plt.show()
    # plt.clf()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, False, b"CasTestSave/LinearTricky3D_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/LinearTricky3D_ML.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("Modèle Linéaire - Linear Tricky 3D")
    # showGraphModeleLineaire(W, False, borne, colors, True)
    #
    # # PMC
    # pmc = CreatePMC([2, 1])
    # TrainPMC(pmc, 10000, 0.01, False)
    # SavePMC(pmc, b"CasTestSave/LinearTricky3D_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/LinearTricky3D_PMC.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("PMC - Linear Tricky 3D")
    # showGraphPMC(pmc, False, borne, len(Y[0]), colors, True)
    # FreePMC(pmc)
    #
    # # # RBF
    # print("RBF")
    #
    # gamma = 0.01
    #
    # uks = X
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/LinearTricky3D_RBF.txt")
    # filename = b"CasTestSave/LinearTricky3D_RBF.txt"
    # W, uks = LoadRBF(filename)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("RBF - Linear Tricky 3D")
    # showGraphRBF(W, False, borne, gamma, uks, len(Y[0]), colors, True)

    # Non Linear Simple 3D
    print("Non Linear Simple 3DD")

    # Création Dataset
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    Y = np.array([
        [2],
        [1],
        [-2],
        [-1]
    ])

    borne = (0, 100)

    colors = ["blue"]

    points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Non Linear Simple 3D")
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    plt.show()
    plt.clf()

    points_c = (c_float * len(points))(*points)
    classes_c = (c_float * len(classes))(*classes)

    # Modele Lineaire
    # W = TrainModeleLineaire(1000, 0.01, False, b"CasTestSave/NonLinearSimple3D_ML.txt")
    #
    # W = LoadModeleLineaire(b"CasTestSave/NonLinearSimple3D_ML.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("Modèle Linéaire - Non Linear Simple 3D")
    # showGraphModeleLineaire(W, False, borne, colors, True)

    # PMC
    # pmc = CreatePMC([2, 2, 1])
    # TrainPMC(pmc, 100000, 0.001, False)
    # SavePMC(pmc, b"CasTestSave/NonLinearSimple3D_PMC.txt")
    # pmc = CreatePMCFromFile(b"CasTestSave/NonLinearSimple3D_PMC.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("PMC - Non Linear Simple 3D")
    # showGraphPMC(pmc, False, borne, len(Y[0]), colors, True)
    # FreePMC(pmc)

    # # RBF
    # print("RBF")
    #
    # gamma = 0.001
    #
    # uks = X
    #
    # W = TrainRBF(gamma, uks, len(uks), b"CasTestSave/NonLinearSimple3D_RBF.txt")
    # filename = b"CasTestSave/NonLinearSimple3D_RBF.txt"
    # W, uks = LoadRBF(filename)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title("RBF - Non Linear Simple 3D")
    # showGraphRBF(W, False, borne, gamma, uks, len(Y[0]), colors, True)
