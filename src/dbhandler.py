import sqlite3
import logging
import os
from typing import Dict, Any

PROJECT_LOGGER = 'ProjectLogger'


class SQLiteProcessor:
    def __init__(self, sqlite):
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.sqlite = sqlite

    def insert(self, table: str, data: Dict[str, Any]):
        self.logger.info(f'Inserting values into {os.path.basename(self.sqlite)}')
        columns = ', '.join(data.keys())
        placeholders = ', '.join('?' for _ in data)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with sqlite3.connect(self.sqlite) as conn:
            conn.executemany(sql, zip(*data.values()))
            conn.commit()

    # def bulk_insert(self, table: str, rows):
    #     if not rows:
    #         return
    #     columns = rows[0].keys()
    #     placeholders = ', '.join('?' for _ in columns)
    #     sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    #
    #     values = [tuple(row[col] for col in columns) for row in rows]
    #
    #     with sqlite3.connect(self.sqlite) as conn:
    #         conn.executemany(sql, values)

    def bulk_insert(self, table: str, rows):

        # Materialize the iterable
        rows = list(rows)
        if not rows:
            return
        columns = rows[0].keys()
        placeholders = ', '.join('?' for _ in columns)
        sql = f"""
            INSERT OR IGNORE INTO {table} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        values = [
            tuple(row[col] for col in columns)
            for row in rows
        ]
        with sqlite3.connect(self.sqlite) as conn:
            conn.executemany(sql, values)

    def bulk_insert_batched(self, table: str, rows, batch_size: int = 5000):
        # To avoid memory problems if data is large
        batch = []

        for row in rows:
            batch.append(row)

            if len(batch) == batch_size:
                self.bulk_insert(table, batch)
                batch.clear()

        if batch:
            self.bulk_insert(table, batch)

    def get_steam_appids(self):
        sql = f"SELECT appid FROM games"
        with sqlite3.connect(self.sqlite) as conn:
            result = conn.execute(sql)
            result = result.fetchall()
        # get the 1st element of the tuple returned by the cursor
        return {el[0] for el in result}

    def update(self, table: str, data: Dict[str, Any], where: str, where_params: tuple = ()):
        set_clause = ', '.join(f"{col} = ?" for col in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        with sqlite3.connect(self.sqlite) as conn:
            conn.execute(sql, tuple(data.values()) + where_params)
            conn.commit()

    def update_rows(self, table: str, data: Dict[str, Any], where: str, where_params: tuple = ()):
        """
        Update records in a table.

        Args:
            table: Table name
            data: Dictionary of column: value pairs to update
            where: WHERE clause (use ? for parameters)
            where_params: Tuple of values for WHERE clause placeholders

        Returns:
            Number of rows affected
        """
        if not data:
            self.logger.warning("No data provided for update")
            return

        set_clause = ', '.join(f"{col} = ?" for col in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"

        try:
            with sqlite3.connect(self.sqlite) as conn:
                cursor = conn.execute(sql, tuple(data.values()) + where_params)
                conn.commit()
                rows_affected = cursor.rowcount
                self.logger.info(f"Updated {rows_affected} row(s) in {table}")
                return rows_affected
        except sqlite3.Error as e:
            self.logger.error(f"Error updating {table}: {e}")
            raise