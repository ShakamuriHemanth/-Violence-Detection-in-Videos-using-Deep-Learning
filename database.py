import sqlite3
import hashlib
import datetime
import MySQLdb
from flask import session
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import numpy as np
 
import os
 
import cv2
import pandas as pd

 
 
 
 

def db_connect():
    _conn = MySQLdb.connect(host="localhost", user="root",
                            passwd="root", db="wpdb")
    c = _conn.cursor()

    return c, _conn

# -------------------------------register-----------------------------------------------------------------
def user_reg(username, email,password):
    try:
        c, conn = db_connect()
        print(username, password, email)
        j = c.execute("insert into user (username,email,password) values ('"+username +
                      "','"+email+"','"+password+"')")
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
    
     
# -------------------------------------Login --------------------------------------
def user_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user where username='" +
                      username+"' and password='"+password+"'")
        data = c.fetchall()
        print(data)     
       
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))


  
if __name__ == "__main__":
    print(db_connect())
