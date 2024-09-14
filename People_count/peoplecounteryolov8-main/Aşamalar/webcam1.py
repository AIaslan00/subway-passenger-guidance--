import sqlite3

# Veritabanı bağlantısı oluşturma
conn = sqlite3.connect('people_counter.db')
cursor = conn.cursor()

# Giriş ve çıkış sayısını tutacak tabloyu oluşturma
cursor.execute('''
CREATE TABLE IF NOT EXISTS people_count (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    entering INTEGER,
    exiting INTEGER
)
''')

conn.commit()
conn.close()


