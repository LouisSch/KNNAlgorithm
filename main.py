# -*- coding: utf-8 -*-
import math
import random as rd
from copy import deepcopy


class Knn:
    def __init__(self, solutionColumn, k, points, variances, maxDist=0.35):
        self.k = k
        self.points = points
        self.variances = variances
        # Si la distance est plus grande, une pénalité sera accordée au point
        self.maxDist = maxDist
        self.solutionColumn = solutionColumn

    def generate_distances(self, points):
        res = dict()
        for i in range(len(self.points)):
            res[i] = []
            for j in range(len(points)):
                if i != j:
                    res[i].append([j, self.points[i].distance_to(points[j], self.variances, self.solutionColumn)])
            res[i] = sorted(res[i], key=lambda x: x[1])
        return res

    def determine_class(self, knnPoints, points):
        # rassemblement des données de décision (classe et distance)
        decisionData = dict()
        # compteur du nombre d'occurences d'une classe
        occurences = dict()
        # score de chaque classe
        score = dict()
        res = ""

        # Ont parcourt de tous les points voisins
        for p, v in knnPoints:
            # On sauvegarde la classe du point et sa distance
            decisionData[p] = [points[p].data[self.solutionColumn], v]
        counter = 0

        # comptage du nombre d'occurences d'une classe
        for v in decisionData.values():
            if v[0] not in occurences:
                occurences[v[0]] = 1
            else:
                occurences[v[0]] += 1

        # Attribution des scores
        if len(occurences) == 1:
            # Choix simple si une seule classe est présente
            res = list(occurences.keys())[0]
        else:
            # Sinon on attribue un score à chaque classe
            for v in decisionData.values():
                if v[0] not in score:
                    score[v[0]] = occurences[v[0]]

                    # Si le point est trop loin, on le pénalise
                    if v[1] > self.maxDist: score[v[0]] -= 2
            maxi = max(list(score.values()))
            for l, s in score.items():
                if s == maxi: res = l

        return res

    def attribute_class(self, points):
        distances = self.generate_distances(points)
        # On parcourt toues les distances des voisins
        for k, v in distances.items():
            # Récupération des k voisins les plus proches du point
            knn = v[:self.k]
            self.points[k].data.append(self.determine_class(knn, points))


def mean(l):
    return sum(l) / len(l)


def mean_on_coordinate(points, coordIndex):
    res = 0
    for el in points:
        res += el.data[coordIndex]
    res /= len(points)
    return res


def calculate_variances(points):
    res = []
    sum = 0
    for i in range(len(points[0].data)):
        mean = mean_on_coordinate(points, i)
        for j in range(len(points)):
            sum += (points[j].data[i] - mean) ** 2
        sum /= len(points)
        res.append(sum)
    return sorted(res, reverse=True)


class KnnPoint:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        dataTemp = map(str, self.data)
        return " | ".join(dataTemp)

    def distance_to(self, point, variances, solutionColumn):
        res = 0
        for k in range(len(self.data)):
            if len(self.data) < (solutionColumn + 1) or k < solutionColumn:
                if variances[k] >= 0.02:
                    res += (point.data[k] - self.data[k]) ** 2
                else:
                    # Si la variance de l'axe est trop faible,
                    # on amplifie la distance de 20% pour
                    # qu'elle ait un meilleur impact dans la
                    # décision finale
                    res += ((point.data[k] - self.data[k]) * 1.2) ** 2
        return math.sqrt(res)


def get_dataset(file, solutionColumn):
    dataSet = []
    with open(file, "r") as file:
        for line in file:
            dataRow = [s.replace("\n", "") for s in line.split(",")]
            if len(dataRow) > 1:
                for k in range(solutionColumn): dataRow[k] = float(dataRow[k])
                dataSet.append(dataRow)
    return dataSet


# Sélectionne une proportion : retourne un tuple avec :
# dataSet1 : dataSet pour le KNN
# dataSet2 : dataSet pour faire la matrice de confusion
# dataSet3 : dataSet pour créer un nuage de point pour associer les points
def select_set_proportion(dataSet, solutionColumn, solutions):
    dataSet1 = []
    dataSet2 = []
    dataSet3 = []
    # Compteur pour dataSet1,dataSet2 et dataSet3
    cpt = [[0 for k in range(len(solutions))], [0 for k in range(len(solutions))]]
    nbrSet = len(dataSet) // 3
    quantity = nbrSet // len(solutions)
    maxIter = len(dataSet)
    i = 0

    # Pour les dataSets 1 et 2
    while any(k < nbrSet for k in cpt[0]) and i < maxIter:
        # Tirage random de l'index de l'élément sélectionné
        elIndex = rd.randint(0, len(dataSet) - 1)
        if cpt[0][solutions.index(dataSet[elIndex][solutionColumn])] < quantity:
            # Ajout d'une copie dans les dataSet
            dataSet1.append(dataSet[elIndex].copy())
            dataSet2.append(dataSet[elIndex].copy())
            cpt[0][solutions.index(dataSet[elIndex][solutionColumn])] += 1

            # Pour éviter de retomber sur les mêmes points
            dataSet.pop(elIndex)

            # On supprime les solutions pour le dataSet 1
            dataSet1[len(dataSet1) - 1].pop(solutionColumn)

        # Pour éviter que ça tourne à l'infini
        i += 1

    i = 0
    while any(k < nbrSet for k in cpt[1]) and i < maxIter:
        # Tirage random de l'index de l'élément sélectionné
        elIndex = rd.randint(0, len(dataSet) - 1)
        if cpt[1][solutions.index(dataSet[elIndex][solutionColumn])] < quantity:
            # Ajout d'une copie dans les dataSet
            dataSet3.append(dataSet[elIndex].copy())
            cpt[1][solutions.index(dataSet[elIndex][solutionColumn])] += 1

            # Pour éviter de retomber sur les mêmes points
            dataSet.pop(elIndex)

        i += 1

    return dataSet1, dataSet2, dataSet3


def initialize_points(dataSet):
    return [KnnPoint(p) for p in dataSet]


def choose_best_k_solution(points, variances, classes, solutionColumn, nbr_iterations):
    best_accuracy = 0
    best_values = []
    k = 1
    # On teste avec différentes valeurs de k
    while nbr_iterations != 0:
        algo = Knn(solutionColumn, k, points[0], variances)
        algo.attribute_class(points[2])
        confMat, accuracy = confusion_matrice(points[0], points[1], classes, solutionColumn)

        if len(best_values) == 0 or accuracy > best_accuracy:
            best_accuracy = accuracy
            best_values = [k, deepcopy(points[0]), best_accuracy, confMat]
        nbr_iterations -= 1
        k += 1

        for p in points[0]:
            del p.data[solutionColumn]

    return best_values


def confusion_matrice(dataKnn, trueData, classes, solutionColumn):
    res = [{c: 0 for c in classes}, {c: 0 for c in classes}]
    for k in range(len(dataKnn)):
        if dataKnn[k].data[solutionColumn] == trueData[k].data[solutionColumn]:
            res[0][dataKnn[k].data[solutionColumn]] += 1
        else:
            res[1][dataKnn[k].data[solutionColumn]] += 1
    accuracy = sum([k for k in res[0].values()]) / len(dataKnn)
    return res, accuracy


def main_program():
    # classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    classes = ['classA', 'classB', 'classC', 'classD', 'classE']
    solutionColumn = 6

    ''' Procédé avec 3 sous-datasets (avec les datasets Iris, data et preTest)
    dataSet = get_dataset("preTest.csv", solutionColumn)
    data1, data2, data3 = select_set_proportion(dataSet, solutionColumn, classes)
    points1, points3 = initialize_points(data1), initialize_points(data2), initialize_points(data3)

    variances = calculate_variances(points1)
    res = choose_best_k_solution([points1, points2], variances, classes, solutionColumn, 10)
    confMat = res[3]
    accuracy = res[2]

    print("--- Valeurs justes")
    for k, v in confMat[0].items():
        print("{key} : {value}".format(key=k, value=v))

    print("--- Valeurs fausses")
    for k, v in confMat[1].items():
        print("{key} : {value}".format(key=k, value=v))
    print("Percentage of accuracy (k choisi : %d): %f" % (res[0], round(accuracy * 100, 2)))
    '''

    d_setTest = get_dataset(r"finalTest.csv", solutionColumn)
    d_set = get_dataset(r"preTest.csv", solutionColumn)

    points1, points2 = initialize_points(d_setTest), initialize_points(d_set)
    variances = calculate_variances(points1)

    # k = 3 est la solution optimale trouvée empiriquement
    algo = Knn(solutionColumn, 3, points1, variances)
    algo.attribute_class(points2)

    # Registering all classes
    with open("result.txt", "w") as f:
        for p in points1:
            f.write(p.data[solutionColumn])
            f.write("\n")

if __name__ == "__main__":
    main_program()
