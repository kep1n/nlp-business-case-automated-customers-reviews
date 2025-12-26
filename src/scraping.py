import os
import json
import logging
import random
import time
import requests
import logclass
import bs4
import utils
import dbhandler

from collections import defaultdict

PROJECT_LOGGER = 'ProjectLogger'


class SteamScraper:
    def __init__(self, html: str):
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.html = html
        self.encoding = 'utf-8'
        self.tag = 'tbody'
        self.data = defaultdict(list)
        self.games_data = dict()
        self.numeric_attribute = 'data-sort'

    def read_html(self) -> bs4.BeautifulSoup:
        try:
            self.logger.info(f'Reading local html: {self.html}...')
            soup = bs4.BeautifulSoup(open(self.html, encoding=self.encoding).read(), features='html.parser')
            return soup
        except Exception as e:
            self.logger.exception(e)

    def get_table_data(self, soup) -> bs4.element.ResultSet:
        """
        Gets the first element of the input tag

        NOTE: We know beforehand that the table we need is the first one
        That's why we use a local html file, to ease things a little bit

        Parameters
        ----------
        soup: parsed html

        Returns
        -------
        bs4.element.ResultSet

        """
        return soup.find_all(self.tag)[0]

    def add_games_data(self, values: list):
        """
        Store games as row-based dictionaries.
        Deduplication enforced by appid.
        """
        fields = ('appid', 'title', 'price', 'rating', 'release_date', 'followers')
        row = dict(zip(fields, values))

        appid = row['appid']

        # Deduplicate (or overwrite, your choice)
        if appid in self.games_data:
            return

        self.games_data[appid] = row

    def process_data(self):
        numerical_attr = self.numeric_attribute
        soup = self.read_html()
        tbody = self.get_table_data(soup)

        self.logger.info('Processing data from html...')

        for tr in tbody.find_all('tr'):
            app_id = tr.get('data-appid')
            tds = tr.find_all('td')

            title = tds[2].find('a').text

            try:
                price = utils.parse_number(tds[4].get(numerical_attr)) / 100
            except TypeError:
                price = None

            rating = utils.parse_number(tds[5].get(numerical_attr))
            release_date = utils.parse_dates(tds[6].get(numerical_attr))
            followers = utils.parse_number(tds[7].get(numerical_attr))

            self.add_games_data([
                app_id,
                title,
                price,
                rating,
                release_date,
                followers
            ])


class SteamAutoReviewer:
    def __init__(self, language: str = 'english', num_per_page: int = 100, review_type: str = 'all'):
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.base_url = 'https://store.steampowered.com/appreviews'
        self.params = {'json': 1, 'language': language, 'cursor': '*', 'num_per_page': num_per_page, 'review_type': review_type}
        self.players_data = dict()
        self.reviews_data = []
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        self.session = requests.Session()

    def iter_users_reviews(self, app_id: int):
        authors = set()
        url = f"{self.base_url}/{app_id}"
        self.logger.info(f'Requesting url: {url}...')
        for review in self._paginate_reviews(url):
            author = review["author"]
            author_id = author["steamid"]

            if author_id not in authors:
                authors.add(author_id)
                yield ("player", [
                    author_id,
                    author["num_reviews"],
                    author["num_games_owned"],
                ])

            yield ("review", [
                app_id,
                author_id,
                review.get("review", None),
                utils.parse_dates(review.get("timestamp_created", None)),
                author.get("playtime_at_review", None),
                author.get("playtime_forever", None),
                review.get("voted_up", None),
                review.get("votes_funny", None),
                review.get("votes_up", None),
                review.get("received_for_free", None),
                review.get("weighted_vote_score", None),
                review.get("steam_purchase", None),
                review.get("comment_count", 0),
                review.get("recommendationid")
            ])

    def _paginate_reviews(self, url: str, max_reviews=1000):
        seen_cursor = None
        self.session.headers.update(self.headers)
        total_yielded = 0
        seen_cursors = set()
        try:
            while True:
                self.logger.info(f'Fetching reviews for cursor {self.params["cursor"]}')
                response = self.session.get(url, params=self.params)
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    # Fallback: decode with utf-8-sig
                    data = json.loads(response.content.decode('utf-8-sig'))
                next_cursor = data.get("cursor")
                if next_cursor in seen_cursors:
                    break
                reviews = data.get("reviews", [])
                if not reviews:
                    break
                # Limit how many reviews we yield
                if max_reviews:
                    remaining = max_reviews - total_yielded
                    reviews = reviews[:remaining]

                yield from reviews
                total_yielded += len(reviews)

                # Stop if we've reached the limit
                if max_reviews and total_yielded >= max_reviews:
                    self.logger.info(f'Reached review limit at pagination level: {total_yielded}')
                    break
                seen_cursors.add(next_cursor)
                self.params["cursor"] = next_cursor
                time.sleep(random.uniform(1.0, 4.0))
            # Restart the cursor
            self.params["cursor"] = '*'
        except Exception as e:
            self.logger.exception(e)

    def fetch_and_store_reviews(self, app_id, db_handler, batch_size=1000):
        """
        Fetch reviews for an app and store them in bulk using the database handler.

        Args:
            app_id: Steam app ID to fetch reviews for
            db_handler: SQLiteProcessor database handler instance with bulk_insert method
            batch_size: Number of records to batch before inserting
        """
        self.logger.info(f'Fetch and store started for {app_id}')
        for record_type, values in self.iter_users_reviews(app_id):
            if record_type == "player":
                self.add_players_data(values)

                # Insert players in batches
                if len(self.players_data) >= batch_size:
                    db_handler.bulk_insert('players', list(self.players_data.values()))
                    self.players_data.clear()

            elif record_type == "review":
                self.add_reviews_data(values)

                # Insert reviews in batches
                if len(self.reviews_data) >= batch_size:
                    db_handler.bulk_insert('reviews', self.reviews_data)
                    self.reviews_data.clear()

        # Insert any remaining records
        if self.players_data:
            db_handler.bulk_insert('players', list(self.players_data.values()))
            self.players_data.clear()
        if self.reviews_data:
            db_handler.bulk_insert('reviews', self.reviews_data)
            self.reviews_data.clear()

    def get_query_summary(self, app_id: int):
        url = f"{self.base_url}/{app_id}"
        try:
            response = self.session.get(url, params=self.params).json()
        except json.JSONDecodeError:
            # Fallback: decode with utf-8-sig
            response = self.session.get(url, params=self.params)
            response = json.loads(response.content.decode('utf-8-sig'))
        query_summary = response['query_summary']
        return {'review_score': query_summary.get('review_score', None),
                'review_score_desc': query_summary.get('review_score_desc', None),
                'total_positive': query_summary.get('total_positive', None),
                'total_negative': query_summary.get('total_negative', None),
                'total_reviews': query_summary.get('total_reviews', None)}

    @staticmethod
    def update_appids_data(appid, table, data, db_handler):
        where = 'appid = ?'
        where_params = (appid,)
        db_handler.update_rows(table, data, where, where_params)

    def add_players_data(self, values: list):
        """
        Store unique players as row-based dictionaries.
        Deduplication is enforced internally by author_id.
        """
        fields = ('author_id', 'num_reviews', 'num_games')
        row = dict(zip(fields, values))

        author_id = row['author_id']

        # Deduplicate by primary key
        if author_id in self.players_data:
            return

        self.players_data[author_id] = row

    def add_reviews_data(self, values: list):
        """
        Store reviews as row-based dictionaries.
        Order is preserved.
        """
        fields = (
            'appid', 'author_id', 'review', 'review_date',
            'playtime_review', 'playtime_forever', 'voted_up',
            'votes_funny', 'votes_up', 'received_free',
            'weighted_vote_score', 'steam_purchase', 'comment_count', 'recommendationid'
        )

        self.reviews_data.append(dict(zip(fields, values)))

    def collect_users_reviews(self, app_id, db):
        for kind, values in self.iter_users_reviews(app_id):
            if kind == "player":
                db.insert(
                    "players",
                    ("author_id", "num_reviews", "num_games"),
                    values,
                )
            elif kind == "review":
                db.insert(
                    "reviews",
                    (
                        "appid", "author_id", "review", "review_date",
                        "playtime_review", "playtime_forever", "voted_up",
                        "votes_funny", "votes_up", "received_free",
                        "weighted_vote_score", "steam_purchase", "comment_count",
                    ),
                    values,
                )


def main():
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'project_log.log')
    logger = logclass.ProjectLogger(log_path=log_path).get_logger()
    logger.info('Process started...')
    # steam_scraper = SteamScraper(
    #     r'E:\SynologyDrive\ironhack\week_14\day_2\project-nlp-business-case-automated-customers-reviews-v2\data\Survival_Horror_SteamDB.html')
    # steam_scraper.process_data()

    # Insert values to the DB
    sqlite_db = dbhandler.SQLiteProcessor(
        r'E:\SynologyDrive\ironhack\week_14\day_2\project-nlp-business-case-automated-customers-reviews-v2\data\gamesDB.db')
    # sqlite_db.bulk_insert("games", steam_scraper.games_data.values())

    # Process reviews
    steam_reviewer = SteamAutoReviewer(review_type='negative')
    # # Insert values to the DB
    for app_id in sqlite_db.get_steam_appids():
    #     data = steam_reviewer.get_query_summary(app_id)
    #     steam_reviewer.update_appids_data(app_id, 'games', data, sqlite_db)
        steam_reviewer.fetch_and_store_reviews(app_id, sqlite_db)
    # logger.info(f'Process finished successfully')


if __name__ == '__main__':
    main()
