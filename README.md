## Traffic Classification
Traffic congestion and related problems are a common concern in urban areas. Understanding traffic patterns and analyzing data can provide valuable insights for transportation planning, infrastructure development, and congestion management. The goal is this project to classify traffic situation using Traffic.csv dataset. The data set contains Time, Date, CarCount, BikeCount, BusCount, TruckCount, Total as features. The Traffic Situation labels are given as low, normal, high, and heavy. To classify the traffic situation k-Nearest Neighbour, Gaussian Naive Bayes, and Support Vector Machine with Gaussian Kernel classifiers is utilized as learning algorithm. All the statistical learning algorithms are implemented from scratch and written in python. 

**Learning Algorithms for Classification**
 - **k-Nearest Neighbour (kNN):** This algorithm classifies data points based on the closest training examples in the feature space. Uses Euclidean/Cosine Distance to find distance between data points.
 - **Gaussian Naive Bayes (NB):** A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
 - **Support Vector Machine with Gaussian Kernel (SVM):** A powerful classifier that works well on a wide range of classification problems, even complex ones. Uses One-vs-Rest Strategy that trains seperate binary classifier for each class, Gradient Descent to find weights and biases for each class.




