import psycopg2

con = psycopg2.connect(database='stackx', user='thomasbuhrmann')
cur = con.cursor()

# Sql statements
# ----------------------------------------
versionSql = "SELECT version()"

def numTagsSql(nrow):
    return "SELECT sum(array_length(regexp_split_to_array(TAGS, '\s'),1)) from RAW_TRAIN WHERE ID < %i;" % (nrow)

def execute(sql):
    cur.execute(sql)
    return cur.fetchone()

# Functions
# ----------------------------------------
def stats():
    print execute(versionSql)
    print execute(numTagsSql(1000))


# ----------------------------------------
if __name__=="__main__":
    stats()