from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets as datasets
import models as models
import numpy as np
import argparse
import locale
import os
import pandas as pd


"""
https://www.pyimagesearch.com/2019/01/21/regression-with-keras/

"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=False, help="path to directory of house dataset",
                default="/Users/patrickryan/Development/python/mygithub/ml_datasets/Houses-Dataset")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# construct a training and testing split with 75% of the data used
# for training and the remaining 25% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (this will lead to
# better training and convergence)
"""
As stated in the comment, scaling our house prices to the range [0, 1] will allow our model to more easily 
train and converge. Scaling the output targets to [0, 1] will reduce the range of our output predictions 
(versus [0, maxPrice ]) and make it not only easier and faster to train our network but enable our model 
to obtain better results as well.
"""
maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
print("[INFO] processing data...")
(trainX, testX) = datasets.process_house_attributes(df, train, test)

# create our MLP and then compile the model using mean absolute
# percentage error as our loss, implying that we seek to minimize
# the absolute percentage difference between our price *predictions*
# and the *actual prices*
model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testX)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
    locale.currency(df["price"].mean(), grouping=True),
    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean error: {:.2f}%, std: {:.2f}%".format(mean, std))

"""
What does this value mean?

Our final mean absolute percentage error implies, that on average, our network will be ~26% 
off in its house price predictions with a standard deviation of ~18%.


"""
