import sqlalchemy
import psycopg2 as pg2
from datetime import datetime
import pandas as pd

conn = pg2.connect(database='postgres', user='postgres', password='134308', host='175.124.190.196', port='5432')

# conn.autocommit = True
cur = conn.cursor()

sql = "select * from dailystockmarketpricedata where stockcode = '005930'"

print(datetime.now())
cur.execute(sql)
rows = cur.fetchall()
conn.commit()
print(datetime.now())
df = pd.DataFrame(rows)

df = df.sort_values(by=1, ascending=False)


