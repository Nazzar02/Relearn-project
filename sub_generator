############################################################################
# Genaration of user's groups and plots reseach of the optimal threshold   #
#                   for the rating of the movies                           #
############################################################################


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------#
#                           GroupGenerator Class                             #
#----------------------------------------------------------------------------#

class GroupGenerator:

    # initialisation of the class

    def __init__(
        self,
        data_path,
        output_path,
        rating_threshold=3.5,
        num_groups=1000,
        group_sizes=range(2, 20),
        min_num_ratings=10,
        train_ratio=0.7,
        val_ratio=0.2,
        negative_sample_size=10,
        verbose=True
    ):
        
    # class attributes

        self.data_path = data_path
        self.output_path = output_path
        self.rating_threshold = rating_threshold
        self.num_groups = num_groups
        self.group_sizes = group_sizes
        self.min_num_ratings = min_num_ratings
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.negative_sample_size = negative_sample_size
        self.verbose = verbose

    # class methods

    # Generate a `users.csv` composed of user id file from `ratings.csv`.
    def generate_users_file(self):
        ratings_path = os.path.join(self.data_path, "ratings.csv")
        ratings = pd.read_csv(ratings_path)
        users = ratings["userId"].drop_duplicates().reset_index(drop=True)
        users_df = pd.DataFrame(users, columns=["userId"])
        users_df.to_csv(os.path.join(self.output_path, "users.csv"), index=False)

    # Generate group ratings from ratings.csv.
    def run(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Generate users.csv if missing
        self.generate_users_file()

        # Generate group ratings and group.csv
        self.generate_group_ratings()

        # Split data into train, val, test
        self.split_data()

        # Visualize average liked titles per group
        self.visualize_liked_titles()


    # Generate group ratings from ratings.csv, storing the average rating (avg_rating).
    def generate_group_ratings(self):
        ratings_path = os.path.join(self.data_path, "ratings.csv")
        ratings = pd.read_csv(ratings_path)

        # Randomly sample groups of users
        user_ids = ratings['userId'].unique()
        np.random.seed(0)
        groups = [
            np.random.choice(user_ids, size=np.random.choice(self.group_sizes), replace=False)
            for _ in range(self.num_groups)
        ]

        group_ratings = []  # (groupId, movieId, avg_rating, timestamp)
        group_data = []     # Composition des groupes

        for group_id, group in enumerate(groups, start=1):
            group_data.append((group_id, list(group)))
            interactions = self._generate_group_interactions(group_id, group, ratings)
            
            # if the list is empty, we take a random user and his ratings
            if not interactions:
                user_with_ratings = np.random.choice(user_ids)
                user_ratings = ratings[ratings['userId'] == user_with_ratings]
                if not user_ratings.empty:
                    for _, row in user_ratings.iterrows():
                        avg_rating = row['rating']  # we stock the rating
                        group_ratings.append((group_id, row['movieId'], avg_rating, row['timestamp']))
                    continue

            group_ratings.extend(interactions)

        # Convert to DataFrame and save
        group_ratings_df = pd.DataFrame(
            group_ratings, 
            columns=["groupId", "movieId", "avg_rating", "timestamp"]
        )
        group_ratings_df.to_csv(os.path.join(self.output_path, "group_ratings.csv"), index=False)
        
        # Save group composition
        group_data_df = pd.DataFrame(group_data, columns=["groupId", "userIds"])
        group_data_df["userIds"] = group_data_df["userIds"].apply(lambda x: str(x))
        group_data_df.to_csv(os.path.join(self.output_path, "group.csv"), index=False)

        print(f"Generated group ratings: {len(group_ratings)} interactions")
        print(f"Group composition saved to group.csv")


    #generate interactions for a group by calculating the average rating for each movie rated by all members of the group.
    def _generate_group_interactions(self, group_id, group, ratings):
        group_ratings = []
        group_movies = {}

        for user_id in group:
            user_ratings = ratings[ratings['userId'] == user_id]
            for _, row in user_ratings.iterrows():
                movie_id = row['movieId']
                rating = row['rating']

                if movie_id not in group_movies:
                    group_movies[movie_id] = []
                group_movies[movie_id].append(rating)

        for movie_id, movie_ratings in group_movies.items():
            if len(movie_ratings) == len(group):  # ensure that all group members rated the movie
                avg_rating = np.mean(movie_ratings)
                timestamp = max(
                    ratings[
                        (ratings['movieId'] == movie_id) & 
                        (ratings['userId'].isin(group))
                    ]['timestamp']
                )
                group_ratings.append((group_id, movie_id, avg_rating, timestamp))

        return group_ratings

    #split group ratings into train, validation, and test datasets.
    def split_data(self):
        group_ratings_path = os.path.join(self.output_path, "group_ratings.csv")
        group_ratings = pd.read_csv(group_ratings_path)

        # Sort by timestamp
        group_ratings = group_ratings.sort_values(by="timestamp").reset_index(drop=True)

        # Split based on ratios
        num_total = len(group_ratings)
        num_train = int(num_total * self.train_ratio)
        num_val = int(num_total * self.val_ratio)

        train_data = group_ratings[:num_train]
        val_data = group_ratings[num_train:num_train + num_val]
        test_data = group_ratings[num_train + num_val:]

        # Save splits
        train_data.to_csv(os.path.join(self.output_path, "group_ratings_train.csv"), index=False)
        val_data.to_csv(os.path.join(self.output_path, "group_ratings_val.csv"), index=False)
        test_data.to_csv(os.path.join(self.output_path, "group_ratings_test.csv"), index=False)

        #chat GPT helped me to get a nice printed output
        print(
            f"Train set size: {len(train_data)}, "
            f"unique groups: {train_data['groupId'].nunique()}, "
            f"unique movies: {train_data['movieId'].nunique()}"
        )
        print(
            f"Validation set size: {len(val_data)}, "
            f"unique groups: {val_data['groupId'].nunique()}, "
            f"unique movies: {val_data['movieId'].nunique()}"
        )
        print(
            f"Test set size: {len(test_data)}, "
            f"unique groups: {test_data['groupId'].nunique()}, "
            f"unique movies: {test_data['movieId'].nunique()}"
        )

    #visualize the average number of liked titles per group and find the optimal rating threshold based on avg_rating.
    def visualize_liked_titles(self):

        group_ratings_path = os.path.join(self.output_path, "group_ratings.csv")
        group_ratings = pd.read_csv(group_ratings_path)

        # 1) find the optimal threshold
        optimal_threshold = self._find_optimal_threshold(group_ratings)

        # 2) Binaerize ratings based on the optimal threshold
        group_ratings['binarized'] = (group_ratings['avg_rating'] >= optimal_threshold).astype(int)

        # 3) Count the total number of liked and disliked titles
        total_liked = group_ratings['binarized'].sum()
        total_disliked = len(group_ratings) - total_liked
        total_interactions = len(group_ratings)

        liked_percentage = (total_liked / total_interactions) * 100 if total_interactions else 0
        disliked_percentage = (total_disliked / total_interactions) * 100 if total_interactions else 0

        # 4) calculate the average number of liked titles per group
        group_stats = group_ratings.groupby('groupId').apply(
            lambda df: df['binarized'].mean()
        )

        # 5) plot the average number of liked titles per group (chat GPT helped me to get a nice plot)
        plt.figure(figsize=(14, 7))
        plt.bar(group_stats.index, group_stats.values, color='skyblue', label='Average Liked Titles per Group')

        plt.axhline(
            y=liked_percentage/100, 
            color='red', 
            linestyle='--', 
            label='Fraction of total liked titles'
        )

        plt.title(
            "Average Number of Liked Titles per Group\n"
            f"Optimal rating threshold = {optimal_threshold:.2f}\n"
            f"Total liked titles: {total_liked} ({liked_percentage:.2f}%)\n"
            f"Total disliked titles: {total_disliked} ({disliked_percentage:.2f}%)"
        )

        plt.xlabel("Group ID")
        plt.ylabel("Fraction Liked Titles (after binarization)")
        plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.output_path, "liked_titles_per_group.png"))
        plt.show()

    def _find_optimal_threshold(self, group_ratings):
        """
        Trouve le seuil qui sépare le plus équitablement possible 
        le nombre de titres likés (avg_rating >= threshold)
        et dislikés (avg_rating < threshold).
        """
        # we go from 0.5 to 5.0 with 100 steps
        rating_range = np.linspace(0.5, 5.0, 100)

        best_threshold = self.rating_threshold
        min_diff = float('inf')

        for threshold in rating_range:
            liked = group_ratings[group_ratings['avg_rating'] >= threshold].shape[0]
            disliked = group_ratings[group_ratings['avg_rating'] < threshold].shape[0]
            diff = abs(liked - disliked)

            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold

            if diff == 0:
                break

        return best_threshold


#----------------------------------------------------------------------------#
#                                 main                                       #
#----------------------------------------------------------------------------#


if __name__ == "__main__":
    data_path = "/home/enzoc/RL/ml-latest-small"
    output_path = "/home/enzoc/RL/ml-latest-small-processed"

    group_generator = GroupGenerator(data_path=data_path, output_path=output_path)
    group_generator.run()
