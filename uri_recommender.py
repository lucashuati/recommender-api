import pandas as pd
from recommendation import Recommendation
from recommender import Recommender

class UriRecommender(Recommender):
    def __init__(self):
        super().__init__()
        self.data_file = 'data_files/solutions_uri.csv'
    
    def load_solutions(self, category=None):
        solutions_df = pd.read_csv(self.data_file)
        if category:
            solutions_df = solutions_df.loc[solutions_df['category_id'] == category]
            solutions_df = solutions_df.loc[solutions_df['solved'] > 100]
        return solutions_df
    
    def run(self, category=None, problems_solved=[]):
        solutions_df = self.load_solutions(category)
        thetas = Recommendation.get_thetas(
            solutions_df,
            nr_features=self.NR_FEATURES,
            nr_epochs=self.NR_EPOCHS,
            nr_iterations=self.NR_ITERATIONS
        )
        # Thetas already calculated for math category(5)
        # thetas = pd.read_csv('data_files/thetas.csv')
        # thetas = thetas.to_dict('list')
        next_problem = Recommendation.get_next_problem(
            problems_solved,
            thetas,
            nr_features=self.NR_FEATURES,
            aggressivity_radius=self.AGGRESSIVITY_RADIUS
        )
        return next_problem
