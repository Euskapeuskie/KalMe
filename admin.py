import sqlite3


if __name__ == "__main__":
    """Test database connection"""
    conn = sqlite3.connect('./feedback.db')
    c = conn.cursor()
    t = c.execute("""SELECT * FROM feedback ORDER BY timestamp DESC""")
    for row in t:
        print(row)
    conn.close()
