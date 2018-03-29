import pymysql
import yaml
import datetime

# MySQL configurations
db = yaml.load(open('db.yaml'))

def getAllRecords():
    try:
        connection = pymysql.connect(host = db['mysql_host'],
                                    user = db['mysql_user'],
                                    password = db['mysql_password'],
                                    db = db['mysql_db'],
                                    cursorclass=pymysql.cursors.DictCursor,
                                    use_unicode=True, charset="utf8")

        cursor = connection.cursor()
        resultVal = cursor.execute("SELECT * from nlpdb order by ID desc;")
        print("rows:",resultVal)

        if resultVal > 0:
            records = cursor.fetchall()
            print(records)

            #connection.close()
            return records
        else:
            print("Kayıt YOK!")

    finally:
        connection.close()

def addRecord(_UserInput,_PredictedText, _IP):
    try:
        connection = pymysql.connect(host = db['mysql_host'],
                                    user = db['mysql_user'],
                                    password = db['mysql_password'],
                                    db = db['mysql_db'],
                                    cursorclass=pymysql.cursors.DictCursor,
                                    use_unicode=True, charset="utf8")

        ID = 0
        cursor = connection.cursor()
        try:
            sql = (
                "INSERT INTO nlpdb (UserInputText, PredictedText, IsTrue, UserSuggestion, CreatedDate, LastModifiedDate, IP) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            )

            cursor.execute(sql, (str(_UserInput), str(_PredictedText), "-1", "", datetime.datetime.now(), datetime.datetime.now(), str(_IP)))

            #connection.commit()
            ID = cursor.lastrowid
            #cursor.close()

        except Exception as e:
            print("error",str(e))
        finally:   
            return ID  

    finally:
        connection.close()

def updateRecord(_ID, _IsTrue, _UserSuggestion):
    try:
        connection = pymysql.connect(host = db['mysql_host'],
                                    user = db['mysql_user'],
                                    password = db['mysql_password'],
                                    db = db['mysql_db'],
                                    cursorclass=pymysql.cursors.DictCursor,
                                    use_unicode=True, charset="utf8")

        ID = 0
        cursor = connection.cursor()
        try:
            sql = (
                "UPDATE nlpdb SET IsTrue = %s, UserSuggestion=%s, LastModifiedDate=%s WHERE ID = %s"
            )

            cursor.execute(sql, (int(_IsTrue), str(_UserSuggestion), datetime.datetime.now(), int(_ID)))

            #connection.commit()
            #cursor.close()
            ID = 1

        except Exception as e:
            print("error",str(e))
        finally:   
            return ID   
    finally:
        connection.close()       