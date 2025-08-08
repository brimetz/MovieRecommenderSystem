def test_basic_math():
    assert 1+1 == 2

from content_based_reco import get_nlp_content_based_recommendations
import pandas as pd
import numpy as np

def test_content_recommendation():
    df = pd.DataFrame({
        'title': ['Toy Story (1995)', 'Jumanji (1995)'],
        'title_norm': ['toy story (1995)', 'jumanji (1995)'],
        'text_features': ['toy story adventure', 'jumanji adventure']
    })
    matrix = np.identity(2)
    result = get_nlp_content_based_recommendations('Toy Story (1995)', matrix, df, top_n=1)
    assert len(result) == 1