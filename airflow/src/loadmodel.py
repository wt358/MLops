
import gridfs
import os
from gridfs import GridFS
import joblib
from bson import ObjectId
from pymongo import MongoClient
import tensorflow.keras.backend as K
import io

class ModelSingleton(type):
   """
   Metaclass that creates a Singleton base type when called.
   """
   _mongo_id = {}
   def __call__(cls, *args, **kwargs):
       mongo_id = kwargs['mongo_id']
       if mongo_id not in cls._mongo_id:
           print('Adding model into ModelSingleton')
           cls._mongo_id[mongo_id] = super(ModelSingleton, cls).__call__(*args, **kwargs)
       return cls._mongo_id[mongo_id]
   
class LoadModel(metaclass=ModelSingleton):
   def __init__(self, *args, **kwargs):
       print(kwargs)
       self.mongo_id = kwargs['mongo_id']
       self.clf = self.load_model()
   def load_model(self):
       print('loading model')

       mongoClient = MongoClient()
       #host = Variable.get("MONGO_URL_SECRET")
       host = os.environ['MONGO_URL_SECRET'] 
       client = MongoClient(host)

       db_model = client['coops2022_model']
       fs = gridfs.GridFS(db_model)
       print(self.mongo_id)
       f = fs.find({"_id": ObjectId(self.mongo_id)}).next()
       print(f)
       with open(f'{f.model_name}.joblib', 'wb') as outfile:
           outfile.write(f.read())
       return joblib.load(f'{f.model_name}.joblib')
   
   
def SaveModel(model,collection_name,model_name,train_dt):
    print('saving model...')
    host = os.environ['MONGO_URL_SECRET']
    client=MongoClient(host)
    db_model = client['coops2022_model']
    fs = gridfs.GridFS(db_model)
    collection_model = db_model[collection_name]
       
    model_fpath = f'{model_name}.joblib'
    joblib.dump(model, model_fpath)

    # save the local file to mongodb
    with open(model_fpath, 'rb') as infile:
        file_id = fs.put(
            infile.read(),
            model_name=model_name
        )
        # insert the model status info to ModelStatus collection
        params = {
            'model_name': model_name,
            'file_id': file_id,
            'inserted_time': train_dt 
        }
        result = collection_model.insert_one(params)
    client.close()
    return result
