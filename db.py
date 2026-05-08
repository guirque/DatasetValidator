from pymongo import MongoClient, UpdateOne
import streamlit as st

connection_string = st.secrets['mongodb']['connection_string']
connection = MongoClient(connection_string)

tcc_data = connection.get_database('TCC_DB')

collection = tcc_data.get_collection('tcc')

def add_registers(objs:list):
    # https://stackoverflow.com/a/54314301/22849468

    update_queries = [UpdateOne({'img':obj['img']}, {'$set':obj}, upsert=True) for obj in objs]
    collection.bulk_write(update_queries)