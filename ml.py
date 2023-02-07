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

script_dir = os.path.abspath(os.path.dirname("CppLib\cmake-build-debug\\"))
lib_path = os.path.join(script_dir, "CppLib.dll")

#print(script_dir, lib_path)

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
savePMC.argtypes = [ctypes.c_void_p]

createPMCFromFile = c_lib.createPMCFromFile
createPMCFromFile.argtypes = [ctypes.c_void_p]
createPMCFromFile.restype = ctypes.c_void_p

saveML = c_lib.saveML
saveML.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]

saveRBF = c_lib.saveRBF
saveRBF.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

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


def TrainModeleLineaire(nb_rep, alpha, is_classification, filename=b"linear_model_save.txt"):
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

def PredictModeleLineaire(input, W, is_classification):
    # Model Lineaire C++
    print("Prediction")
    if (len(Y[0]) > 2):
        maxMat = [np.matmul(np.transpose(Wt), np.array([1.0, *input])) for Wt in W]
        index = maxMat.index(max(maxMat))

        print(list(reversed(sorted([(maxMat[i], results[i]) for i in range(len(maxMat))], key=lambda x: x[0],))))
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
    if len(Y[0]) > 2:
        W = []
        for i in range(yCol):
            W.append(res[i * size:i * size + size])
    else:
        W = res[:size]
    return W

def showGraphLinear(W, is_classification, borne):
    if is_classification:
        test_points = []
        test_colors = []
        for row in range(borne[0], borne[1]):
            for col in range(borne[0], borne[1]):
                p = np.array([col / 100, row / 100])
                c = PredictModeleLineaire(p, W, is_classification)
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
        plt.scatter(X[:, 0], X[:, 1], c=colors)
    else:
        plt.scatter(X, Y)
        fitline = predict(W[0], X, 0)
        plt.plot(X, fitline, c="r")
    plt.show()

def predict(slope, x, intercept):
    return slope * x + intercept

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

def PredictPMC(pmc, input, is_classification):
    input_c = (c_float * len(input))(*input)
    t = predictPMC(pmc, input_c, is_classification)

    if (len(Y[0]) > 2):
        maxMat = [t[i] for i in range(len(Y[0]))]
        print(list(reversed(sorted([(maxMat[i], results[i]) for i in range(len(maxMat))], key=lambda x: x[0], ))))
        index = maxMat.index(max(maxMat))
        c = results[index]
    else:
        c = 'pink' if t[0] <= 0 else 'lightcyan'


    return c

def FreePMC(pmc):
    freeMemory(pmc)

def showGraphPMC(pmc, is_classification, borne):
    if is_classification:
        test_points = []
        test_colors = []
        for row in range(borne[0], borne[1]):
            for col in range(borne[0], borne[1]):
                p = np.array([col / 100, row / 100])
                p_c = (c_float * len(p))(*p)
                c = PredictPMC(pmc, p, is_classification)
                test_points.append(p)
                test_colors.append(c)
        test_points = np.array(test_points)
        test_colors = np.array(test_colors)

        plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
        plt.scatter(X[:, 0], X[:, 1], c=colors)

    else:
        t = predictPMC(pmc, points_c, len(X), len(X[0]), True)
        fitline = t[0] * X + X[0]
        plt.scatter(X, Y)
        plt.plot(X, fitline, c="r")
    plt.show()

def GetKMeans(k):
    uks = []
    kMeanMat = kMean(points_c, k, len(X), len(X[0]))
    for k in range(k):
        uks.append(kMeanMat[k * len(X[0]): k * len(X[0]) + len(X[0])])

    return uks

def TrainRBF(gamma, uks):


    sigma = []

    for i in range(len(X)):
        sigma.append([])
        for k in range(len(results)):
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
    SaveRBF(b"rbf_save.txt", Wc, uks, len(W[0]), len(W))
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

def PredictRBF(input, W, is_classification, gamma, uks):
    # Model Lineaire C++
    print("Prediction RBF")
    if (len(Y[0]) > 2):
        maxMat = []
        for k in range(len(results)):
            out = 0
            for n in range(len(W)):
                uk = uks[n]
                xk = np.array([a - b for a, b in zip(input, uk)])
                norm = sum([elem ** 2 for elem in xk])
                gauss = math.exp(-gamma * norm)
                out += W[n][k] * gauss
            maxMat.append(out)

        index = maxMat.index(max(maxMat))

        print(list(reversed(sorted([(maxMat[i], results[i]) for i in range(len(maxMat))], key=lambda x: x[0],))))
        c = results[index]
        # print(maxMat[results.index("France")])
        # print(maxMat[index])
    else:
        c = 'lightcyan' if np.matmul(np.transpose(W), np.array([*input])) >= 0 else 'pink'
    return c

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

    return [r,g,b]

def averageRGB100(image):
    l10 = []
    ln = [0] * 10
    for k in range(10):
        start = int(image.size[0]/10 * k)
        end = int(image.size[0]/10 * (k+1)) if int(image.size[0]/10 * (k+1)) < image.size[0] else image.size[0]

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

    imX = []
    imY = []
    index = 0
    for foldername in glob.glob('dataset/flag/training/old/*'):
        for filename in glob.glob(foldername + '/*'):
            im = Image.open(filename)
            im = im.convert("RGBA")
            imX.append(averageRGB100(im))
            temp = [-1] * (len([name for name in os.listdir('dataset/flag/training/old/')]))
            temp[index] = 1
            imY.append(temp)
        index += 1

    # Flag dataset
    X = np.array(
        imX
    )

    Y = np.array(
        imY
    )

    # france1 = Image.open('dataset/flag/training/old/france1.png')
    # france1 = france1.convert("RGB")
    # france2 = Image.open('dataset/flag/training/old/france2.jpg')
    # france2 = france2.convert("RGB")
    #
    # usa = Image.open('dataset/flag/training/old/usa1.png')
    # usa = usa.convert("RGB")
    #
    # italie = Image.open('dataset/flag/training/old/italie1.png')
    # italie = italie.convert("RGB")
    #
    # X = np.array(
    #     [
    #         averageRGB100(france1),
    #         averageRGB100(france2),
    #         averageRGB100(usa),
    #         averageRGB100(italie)
    #     ]
    # )
    #
    # Y = np.array(
    #     [
    #         [1, -1, -1],
    #         [1, -1, -1],
    #         [-1, 1, -1],
    #         [-1, -1, 1],
    #     ]
    # )

    france_test = Image.open('dataset/flag/training/fr.png')
    france_test = france_test.convert("RGBA")

    italie_train = Image.open('dataset/flag/training/it.png')
    italie_train = italie_train.convert("RGBA")

    italie_test = Image.open('dataset/flag/afs.jpg')
    italie_test = italie_test.convert("RGBA")

    results = [
        "Andorre",
        "Emirats Arabe Unis",
        "Afghanistan",
        "Antigua et Barbuda",
        "Anguilla",
        "Albanie",
        "Arménie",
        "Angola",
        "Antarctique",
        "Argentine",
        "Samoa américain",
        "Autriche",
        "Australie",
        "Aruba",
        "Aaland",
        "Azerbaidjan",
        "Bosnie-Herzegovine",
        "Barbades",
        "Bangladesh",
        "Belgique",
        "Burkina Faso",
        "Bulgarie",
        "Bahrein",
        "Burundi",
        "Benin",
        "Saint-Barthélemy",
        "Bermude",
        "Brunei",
        "Bolovie",
        "Antilles néerlandaise",
        "Brésil",
        "Bahamas",
        "Bhoutan",
        "Botswana",
        "Biélorussie",
        "Bélize",
        "Canada",
        "Iles Cocos",
        "République Démocratique du Congo",
        "Centrafrique",
        "Congo",
        "Suisse",
        "Cote d'Ivoire",
        "Iles Cook",
        "Chili",
        "Cameroun",
        "Chine",
        "Colombie",
        "Costa Rica",
        "Cuba",
        "Cap Vert",
        "Curaçao",
        "Iles Christmas",
        "Chypre",
        "République Tchèque",
        "Allemagne",
        "Djibouti",
        "Danemark",
        "Dominique",
        "République Dominicaine",
        "Algérie",
        "Equateur",
        "Estonie",
        "Egypte",
        "Sahara Oriental",
        "Erythrée",
        "Espagne",
        "Ethiopie",
        "Finlande",
        "Fidji",
        "Iles Falkland",
        "Micronésie",
        "Iles féroé",
        "France",
        "Gabon",
        "Royaume-Uni",
        "Angleterre",
        "Irlande du Nord",
        "Ecosse",
        "Pays de Galles",
        "Grenade",
        "Géorgie",
        "Guyane",
        "Guernesey",
        "Ghana",
        "Gibraltar",
        "Groenland",
        "Gambie",
        "Guinée",
        "Guadeloupe",
        "Guinée équatoriale",
        "Grèce",
        "Géorgie du Sud",
        "Guatemala",
        "Guam",
        "Guinée-Bissau",
        "Guyana",
        "Hong Kong",
        "Iles Heard et Iles McDonald",
        "Honduras",
        "Croatie",
        "Haïti",
        "Honduras",
        "Indonésie",
        "Irlande",
        "Israel",
        "Ile de Man",
        "Inde",
        "Territoire britannique de l'Ocean Indien",
        "Irak",
        "Iran",
        "Islande",
        "Italie",
        "Jersey",
        "Jamaique",
        "Jordanie",
        "Japon",
        "Kenya",
        "Kyrgyzistan",
        "Cambodge",
        "Kiribati",
        "Comores",
        "Saint Kitts et Nevis",
        "Corée du Nord",
        "Corée du Sud",
        "Koweït",
        "Iles Cayman",
        "Kazakhstan",
        "Laos",
        "Liban",
        "Sainte Lucie",
        "Liechteinstein",
        "Sri Lanka",
        "Libéria",
        "Lesotho",
        "Lituanie",
        "Luxembourg",
        "Lettonie",
        "Libye",
        "Maroc",
        "Monaco",
        "Moldavie",
        "Montenegro",
        "Madagascar",
        "Iles Marshall",
        "Macédoine du Nord",
        "Mali",
        "Birmanie",
        "Mongolie",
        "Macao",
        "Iles Marianne du Nord",
        "Martinique",
        "Mauritanie",
        "Montserrat",
        "Malte",
        "Maurice",
        "Maldives",
        "Malawi",
        "Mexique",
        "Malaysie",
        "Mozambique",
        "Namibie",
        "Nouvelle Calédonie",
        "Niger",
        "Iles Norfolk",
        "Nigeria",
        "Nicaragua",
        "Pays-Bas",
        "Norvège",
        "Népal",
        "Nauru",
        "Niue",
        "Nouvelle Zélande",
        "Oman",
        "Panama",
        "Pérou",
        "Polynésie Française",
        "Papouasie Nouvelle Guinée",
        "Philipines",
        "Pakistan",
        "Pologne",
        "Saint Pierre et Miquelon",
        "Iles Pitcairn",
        "Porto Rico",
        "Palestine",
        "Portugal",
        "Palaos",
        "Paraguay",
        "Qatar",
        "La Réunion",
        "Roumanie",
        "Serbie",
        "Russie",
        "Rwanda",
        "Arabie Saoudite",
        "Iles Solomon",
        "Seychelles",
        "Soudan",
        "Suède",
        "Singapour",
        "Saint Helène",
        "Slovénie",
        "Slovaquie",
        "Sierra Leone",
        "Saint Marin",
        "Sénégal",
        "Somalie",
        "Suriname",
        "Sud Soudan",
        "Sao Tomé-et-Principe",
        "El Salvador",
        "Sint Maarten",
        "Syrie",
        "Eswatini",
        "Iles Turks et Caicos",
        "Tchad",
        "Territoire antarctique français",
        "Togo",
        "Thaïlande",
        "Tadjikistan",
        "Tokelau",
        "Timor Leste",
        "Turkmenistan",
        "Tunisie",
        "Tonga",
        "Turquie",
        "Trinite et Tobago",
        "Tuvalu",
        "Taiwaïn",
        "Tanzanie",
        "Ukraine",
        "Ouganda",
        "Etats-Unis d'Amérique",
        "Uruguay",
        "Ouzbekistan",
        "Vatican",
        "Saint-Vincent et les Grenadines",
        "Venezuela",
        "Iles Vierges Britanniques",
        "Iles Vierges Américaines",
        "Vietnam",
        "Vanuatu",
        "Wallis et Futuna",
        "Samoa",
        "Kosovo",
        "Yemen",
        "Mayotte",
        "Afrique du Sud",
        "Zambie",
        "Zimbabwe",
    ]

    results = sorted(["France", "USA", "Italie", "Espagne", "Allemagne", "Afrique du Sud"])

    points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]

    points_c = (c_float * len(points))(*points)
    classes_c = (c_float * len(classes))(*classes)

    # print("RBF")
    #
    # gamma = 0.0000001
    #
    # uks = GetKMeans(len(results))
    #
    # # W = TrainRBF(gamma, uks)
    # filename = b"rbf_save.txt"
    # W, uks = LoadRBF(filename)
    #
    #
    # print(PredictRBF(averageRGB100(france_test), W, True, gamma, uks))
    # print(PredictRBF(averageRGB100(italie_train), W, True, gamma, uks))
    # print(PredictRBF(averageRGB100(italie_test), W, True, gamma, uks))

    # W_flag = TrainModeleLineaire(100000, 0.001, True)
    # print("Modele Lineaire")
    # filename = b"linear_model_save.txt"
    # W_flag = LoadModeleLineaire(filename)
    # print(PredictModeleLineaire(averageRGB100(italie_train), W_flag, True))
    # print(PredictModeleLineaire(averageRGB100(italie_test), W_flag, True))
    # print(PredictModeleLineaire(averageRGB100(france_test), W_flag, True))

    # print("PMC")
    # pmc_flag = CreatePMC([len(X[0]), 20, 10, len(Y[0])])
    #
    # checkPMC(pmc_flag)
    # pmc_flag = TrainPMC(pmc_flag, 10000, 0.001, True)
    #
    # filename = b"test.txt"
    # SavePMC(pmc_flag, filename)
    # freeMemory(pmc_flag)
    # exit()

    #checkPMC(pmc_flag)
    # print('PMC')
    # filename = b"test.txt"
    # pmc_flag = CreatePMCFromFile(filename)
    #
    # print(PredictPMC(pmc_flag, averageRGB100(france_test), True))
    # print(PredictPMC(pmc_flag, averageRGB100(italie_test), True))
    # print(PredictPMC(pmc_flag, averageRGB100(italie_train), True))
    #
    # freeMemory(pmc_flag)

    # X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    # Y = np.array([[1], [1], [-1], [-1]])
    #
    # borne = (0, 100)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # print(points, classes, colors)
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # pmc_flag = CreatePMC([2, 2, 1])
    # print(pmc_flag)
    #
    # TrainPMC(pmc_flag, 10000, 0.01, True)
    #
    # try:
    #     showGraphPMC(pmc_flag, True)
    # except:
    #     FreePMC(pmc_flag)

    # Linear Simple
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
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # PredictModeleLineaire(1000, 0.01, True)
    #
    # # PMC
    # PredictPMC([2, 1], 10000, 0.01, True)

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

    # Linear Multiple
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
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # #PredictModeleLineaire(10000, 0.01)
    #
    # # PMC
    # #PredictPMC([2, 1], 10000, 0.01)
    #
    # # XOR
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
    #
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # W = TrainModeleLineaire(10000, 0.01, True)
    #
    # showGraphLinear(W, True)
    #
    # # PMC
    # pmcXOR = CreatePMC([2, 3, 1])
    #
    # TrainPMC(pmcXOR, 1000000, 0.01, True)
    #
    # PredictPMC(pmcXOR, [0.2,0.2], True)
    # showGraphPMC(pmcXOR, True)

    #
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
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # #PredictModeleLineaire(10000, 0.01)
    #
    # # PMC
    # #PredictPMC([2, 4, 1], 10000, 0.01)
    #
    # # MultiLinear 3 classes
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
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # # Modele Lineaire
    # #PredictModeleLineaire(10000, 0.01)
    #
    # # PMC
    # #PredictPMC([2,3], 10000, 0.01)
    #
    # # Multi Cross
    # print("Multi Cross")
    #
    # # Création Dataset
    # X = np.random.random((1000, 2)) * 2.0 - 1.0
    # Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
    #     p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])
    #
    # borne = (-100, 100)
    #
    # colors = ["blue" if elem > 0 else "red" for elem in Y] if len(Y[0]) < 2 else [
    #     "blue" if elem[0] == 1 else "red" if elem[1] == 1 else "green" for elem in Y]
    #
    # points = [X[i, j] for i in range(len(X)) for j in range(len(X[i]))]
    # classes = [i for i in Y] if len(Y[0]) < 2 else [Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))]
    #
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
    #
    # points_c = (c_float * len(points))(*points)
    # classes_c = (c_float * len(classes))(*classes)
    #
    # Modele Lineaire
    # PredictModeleLineaire(10000, 0.01)

    # PMC
    # PredictPMC([2, 20, 21, 3], 10000, 0.01)
