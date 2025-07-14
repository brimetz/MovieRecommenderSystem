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

    # filters
    filtered_corr = filtered_corr.drop(index=target_title, errors="ignore")
    filtered_corr.reset_index(inplace=True)
    filtered_corr.rename(columns={"index": "title"}, inplace=True)

    return filtered_corr.head(top_n)