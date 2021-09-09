from app.database import engine
import codecs
import pickle
import zipfile
import os
import tensorflow as tf

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tf_model(model_name):
    """
    * DB에 있는 텐서플로우 모델을 불러옵니다. 
    * 모델은 zip형식으로 압축되어 binary로 저장되어 있습니다.
    * 모델의 이름을 받아 압축 해제 및 tf_model폴더 아래에 저장한 후 로드하여 
      텐서플로우 모델 객체를 반환합니다.
    """
    query = f"""SELECT model_file 
                FROM model_core 
                WHERE model_name='{model_name}';"""
    
    bin_data = engine.execute(query).fetchone()[0]
    print("query")
    
    model_buffer = pickle.loads(codecs.decode(bin_data, "base64"))
    print("buffer")
    model_path = os.path.join(base_dir, "tf_model", model_name)
    print("model path:", model_path)
    with zipfile.ZipFile(model_buffer, "r") as bf:
        bf.extractall(model_path)
    print("unzip")
    tf_model = tf.keras.models.load_model(model_path)
    print("tf_model")

    return tf_model

    # return model_path

