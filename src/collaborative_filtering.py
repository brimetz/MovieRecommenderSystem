import pandas as pd

def get_similar_movies_pearson(target_title, user_movie_matrix, min_common_ratings=10, top_n=10):
    """
    will recommend movie similar to `target_title` using the pearson correlation

    Args:
        target_title (string): movie choose by the user
        user_movie_matrix (panda dataframe): Matrix Users x movies
        min_common_ratings (integer): minimal threshold of shared ratings
        top_n (integer): number of movies to recommend
    
    Returns:
        panda dataframe: movie array with their correlation scoring
    """
    if (target_title not in user_movie_matrix.columns):
        return pd.DataFrame(columns=["title", "correlation", "num_common_ratings"])
    
    # scoring vectors for the target movie
    target_ratings = user_movie_matrix[target_title]

    # correlation compute with other movies
    correlations = user_movie_matrix.corrwith(target_ratings)

    # create a dataframe with the result
    corr_df = pd.DataFrame(correlations, columns=["correlation"])
    corr_df.dropna(inplace=True)

    # add shared scoring number to filter movies lesser score
    num_common = user_movie_matrix.apply(lambda x: target_ratings.notna() & x.notna(), axis=0).sum()
    corr_df["num_common_ratings"] = num_common

    corr_df = corr_df[corr_df["num_common_ratings"] >= min_common_ratings].copy()

    # delete target movie from the result dataframe
    if target_title in corr_df.index:
        corr_df = corr_df.drop(index=target_title)

    # filters and rename
    corr_df = corr_df.sort_values(by="correlation", ascending=False)
    corr_df.reset_index(inplace=True)
    corr_df.rename(columns={"index": "title"}, inplace=True)

    return corr_df.head(top_n)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_similar_movies_cosine(target_title, user_movie_matrix, min_common_ratings=10, top_n=10):
    """
    Recommand similar movies of 'target_title' using the cosine similarity

    Args:
        target_title (string): movie chosen by the user
        user_movie_matrix (panda dataframe): Users x movies matrix
        min_common_ratings (integer): minimal numbers of shared ratings
        top_n (integer): number of movie to return

    Returns:
        pd.Dataframe: movie array with similarity score
    """
    if target_title not in user_movie_matrix.columns:
        return pd.DataFrame(columns=["title", "similarity", "num_common_ratings"])
    
    # Transpose movie matrix
    movie_matrix = user_movie_matrix.T
    # if target movie has no ratings we stop
    if movie_matrix.loc[target_title].isna().all():
        return pd.DataFrame(columns=["title", "similarity", "num_common_ratings"])

    # replace NaN value by O
    filled = movie_matrix.fillna(0)
    sim_mat = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim_mat,
                          index=filled.index,
                          columns=filled.index)

    # remove the movie target
    sim_scores = sim_df[target_title].drop(target_title)

    # Compute shared ratings with other movies
    target_ratings = movie_matrix.loc[target_title]
    num_common = movie_matrix.apply(lambda x: (target_ratings.notna() & x.notna()).sum(),
                                    axis=1).drop(target_title)

    # join each array to be sure to have the same number of element
    result = pd.DataFrame({
        "similarity": sim_scores,
        "num_common_ratings": num_common
    })

    # filters and sorting
    result = result[result["num_common_ratings"] >= min_common_ratings]
    result = result.sort_values(by="similarity", ascending=False)

    result = result.reset_index().rename(columns={"index": "title"})
    return result.head(top_n)