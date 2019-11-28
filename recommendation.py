import pandas as pd
import numpy as np
df = pd.read_csv('Vac_responses.csv')
R_df = df.pivot(index='User', columns='Places', values='rating').fillna(0)
R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 2)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
print(df)
#already_rated, predictions = recommend_movies(preds_df, 1, movies_df, ratings_df, 10)

print(preds_df)