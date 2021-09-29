import random

#In this part we are reading the diabetes.csv file and making the dataset
with open('diabetes.csv', 'r') as iris:
    file = iris.readlines()
    dataset = []
    for a in file:
        dataset.append(a.split(','))
    random.shuffle(dataset)

    #In this part we are spliting train,validation and test set
    train_set = []
    val_set = []
    test_set = []

    for a in dataset:
        num = random.random()
        if 0 <= num <= 0.7:
            train_set.append(a)
        elif 0.7 < num <= 0.85:
            val_set.append(a)
        else:
            test_set.append(a)

    #In this part we are doing the knn regression for val_set
    k = 10
    error = 0
    for v in val_set:

        #here we are calculating eucliadean distance and from that distance we have taken k number of values
        L = []
        for t in train_set:
            distance = ((float(t[0]) - float(v[0]))**2 + (float(t[1]) - float(v[1]))**2 + (float(t[2]) - float(v[2]))**2 + (float(t[3]) - float(v[3]))**2 + (float(t[4]) - float(v[4]))**2 + (float(t[5]) - float(v[5]))**2 + (float(t[6]) - float(v[6]))**2 + (float(t[7]) - float(v[7]))**2 + (float(t[8]) - float(v[8]))**2 + (float(t[9]) - float(v[9]))**2)**0.5
            copy_t = []
            for a in t:
                copy_t.append(a)
            copy_t.append(distance)
            L.append(copy_t)
        L = sorted(L, key=lambda l: l[11])
        L = L[0:k]

        #here we will calculate the average output of k sample
        sum = 0
        for l in L:
            sum += float(l[10])
        #it is the determined output
        average = sum/k

        #Here we will calculate error
        error += (float(v[10]) - average)**2

    #here we will print the mean square error
    mean_square_error = error/len(val_set)
    print("Validation set")
    print("------------------")
    print("For K = {} The mean square error is {} \n\n".format(k, mean_square_error))

    # In this part we are doing the knn regression for test_set
    error = 0
    for v in test_set:

        # here we are calculating eucliadean distance and from that distance we have taken k number of values
        L = []
        for t in train_set:
            distance = ((float(t[0]) - float(v[0])) ** 2 + (float(t[1]) - float(v[1])) ** 2 + (
                    float(t[2]) - float(v[2])) ** 2 + (float(t[3]) - float(v[3])) ** 2 + (
                                    float(t[4]) - float(v[4])) ** 2 + (float(t[5]) - float(v[5])) ** 2 + (
                                    float(t[6]) - float(v[6])) ** 2 + (float(t[7]) - float(v[7])) ** 2 + (
                                    float(t[8]) - float(v[8])) ** 2 + (float(t[9]) - float(v[9])) ** 2) ** 0.5
            copy_t = []
            for a in t:
                copy_t.append(a)
            copy_t.append(distance)
            L.append(copy_t)
        L = sorted(L, key=lambda l: l[5])
        L = L[0:10]

        # here we will calculate the average output of k sample
        sum = 0
        for l in L:
            sum += float(l[10])
        # it is the determined output
        average = sum / 10

        # Here we will calculate error
        error += (float(v[10]) - average) ** 2

    # here we will print the mean square error
    mean_square_error = error / len(test_set)
    print("Test set")
    print("------------------")
    print("For K = 10 The mean square error is {} \n\n".format(mean_square_error))
