
import sqlite3
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()
cursor.execute("SELECT id, symbol FROM assets LIMIT 10")
print(cursor.fetchall())
conn.close()
