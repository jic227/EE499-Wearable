KMeans

I ran kmeans clustering on daily Fitbit features including steps, calories, intensity, and METs. I used k = 4 since there are 4 subjects. The algorithm grouped days into clusters based on activity behavior. The clusters showed different patterns of activity levels across days.

KNN

I used knn to classify daily feature vectors by subject ID. I performed an 80/20 train/test split after shuffling the data. The model achieved about 50% accuracy. This shows that daily activity patterns contain some subject-specific structure, but the classes overlap.

Change Point Analysis (CPA)

I ran CPA on the multiyear dailySteps dataset. The algorithm detected 8 change points clustered in late July to mid August 2013. This suggests a shift in behavior during that time period, possibly seasonal or lifestyle related
