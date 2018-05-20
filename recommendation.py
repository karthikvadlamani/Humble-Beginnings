import numpy as np
from lightfm.datasets import fetch_movielens #fetch_movielens is a method
from lightfm import LightFM #importing lightfm class to create a model later
from lightfm.evaluation import precision_at_k

#fetch the data and format it
data = fetch_movielens(min_rating = 4.0) #limiting movies below rating 4.0

#This method creates an interaction matrix from the data.
#The data has movie names and ratings of it from ALL users

#print(repr(data['train']))
#print(repr(data['test']))

#Create model using the lightfm model, pass the value 'warp' to the loss parameter

#Loss means the loss function and it measures the diff between the model's prediction and the actual output

#warp means Weighted Approximate-Rank Pairwise. It helps us create recommendations
#for each user by looking at the existing user-rating pairs and predicting
#rankings for each. It uses the gradient descent algorithm to iteratively
#find the weights that improve our prediction over time. It takes into
#account a user's past ratings and similar user's ratings for the same title
#thus Content+Collaborative

model_1 = LightFM(loss = 'warp')
model_1.fit(data['train'], epochs=30, num_threads=2)
test_precision_1 = precision_at_k(model_1, data['test'], k=3).mean()

model_2 = LightFM(loss = 'warp-kos',n=10,k=5)
model_2.fit(data['train'], epochs=30, num_threads=2)
test_precision_2 = precision_at_k(model_2, data['test'], k=3).mean()

model_3 = LightFM(loss = 'logistic')
model_3.fit(data['train'], epochs=30, num_threads=2)
test_precision_3 = precision_at_k(model_3, data['test'], k=3).mean()

index = np.argmax([test_precision_1,test_precision_2,test_precision_3])
models = {0:model_1, 1:model_2, 2:model_3}

user_id = int(input("Enter user_id: "))

#We want to generate a recommendation from our model using a sample function
#It takes our model, data and a list of user_ids(For whom we want
#recommendations for)as inputs

def sample_recommendation(model, data, user_id):

	#num of users and movies in training data
	n_users, n_items = data['train'].shape

	#for loop to iterate through every user_id we input and generate
	#recommendations. Get known positives for each user i.e., all movies
	#for which they've given a 5 rating. Rest all are negative.

	known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
		#data['train'].tocsr()[user_id].indices is a subarray in the data matrix that
		#we retrieve using the .indices attribute

		#Movies our model predicts they will like
	scores = model.predict(user_id, np.arange(n_items))
		#Rank them in order of most liked to least
	top_items = data['item_labels'][np.argsort(-scores)]

		#print out the results
	print("User %s" % user_id)
	print("		Known Positives:")

	for x in known_positives[:3]:
		print("			%s" % x)

	print("		Recommended:")

	for x in top_items[:3]:
		print("			%s" % x)

sample_recommendation(models[index], data, user_id)
