from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_house_attributes(inputPath):
    """
    Read in the housing data.  The intial dataset contains 535 rows, but because some zipcodes only
    have a few houses, we are removing rows (houses) in a zipcode if there are less than 25 houses in
    a zipcode.  That will reduce our dataset to 362 rows (houses)
    :param inputPath:
    :return:
    """
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

    # determine (1) the unique zip codes and (2) the number of data
    # points with each zip code
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of the unique zip codes and their corresponding
    # count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zip code counts for our housing dataset is *extremely*
        # unbalanced (some only having 1 or 2 houses per zip code)
        # so let's sanitize our data by removing any houses with less
        # than 25 houses per zip code
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    # return the data frame
    return df


def process_house_attributes(df, train, test):
    """
    Preprocess some of the housing data.
    1) for the continuous values, scale them between 0 and 1
    2) One-Hot encode categorical values with LabelBinarizer (Binarize labels in a one-vs-all fashion)

    When this function completes all features (continous, categorical) are in the range [0,1]

    :param df:
    :param train:
    :param test:
    :return:
    """
    # initialize the column names of the continous data
    continuous = ['bedrooms', 'bathrooms', 'area']

    # perform min-max scalling each continous feature column to the range [0,1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.fit_transform(test[continuous])

    # one-hot encode the zip code categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    # NOTE: we fit on the entirety of the dataset, then transform the training and test
    # because it is possible that the training and test might not have some of the total possible
    # zipcode values.
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX)


def load_house_images(df, inputPath):
    """
    For each index, read the set of 4 images, and creae a 4x4 montage of all separate images
    to create a single image for each house to train on.
    :param df:
    :param inputPath:
    :return:
    """
    # initialize our images array
    images = []

    # loop over the indexes of the house
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialize our list of input images along with the output image
        # after *combining* the four input images
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")

        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)

        # tile the four input images in the output image such the first
        # image goes in the top-right corner, the second image in the
        # top-left corner, the third image in the bottom-right corner,
        # and the final image in the bottom-left corner
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]

        # add the tiled image to our set of images the network will be
        # trained on
        images.append(outputImage)

    # return our set of images
    return np.array(images)


if __name__ == '__main__':
    path = '/Users/patrickryan/Development/datasets/Houses-dataset/Houses-Dataset/HousesInfo.txt'
    df = load_house_attributes(path)
    print(df.shape)
    print(df.head())
    X = df.drop(columns=['price'])
    y = df['price']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    trainX, testX = process_house_attributes(df, X_train, X_test)
    print(trainX)
