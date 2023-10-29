import tensorflow as tf
import json
import os
import struct
import numpy as np

tf.config.list_physical_devices('GPU')
def fit_strokes(points):
    minx = maxx = points[0][0][0]
    miny = maxy = points[0][1][0]
    for stroke in points:
        x,y = stroke
        minx = min(min(x),minx)
        miny = min(min(y),miny)
        maxx = max(max(x),maxx)
        maxy = max(max(y),maxy)
    if max(maxx-minx,maxy-miny) != 0:
        scale = 254.0/max(maxx-minx,maxy-miny)
    arr = []
    for stroke in points:
        x,y = stroke
        stroke = [[int((a-minx)*scale+1),int((b-miny+1)*scale+1)] for a,b in zip(x,y)]
        if len(stroke) > 20:
            newarr = []
            index = 0.0
            for i in range(20):
                newarr.append(stroke[int(index)])
                index += len(stroke)/20.0
            stroke = newarr
        else:
            stroke += [[0,0]]*(20-len(stroke))
        arr.append(stroke)
    if len(arr) > 10:
        arr = arr[:10]
    while len(arr) < 10:
        arr.append([[0,0]]*20)
    return np.array(arr,dtype="float32")

def unpack_drawing(category, file_handle,give_category=False,categories=[]):
    data = file_handle.readline()
    data = json.loads(data)
    if give_category:
        stuff = [0]*15
        stuff[categories.index(category)] = 1
        return stuff
    return fit_strokes(data['strokes'])

def parse_data_from_file(filename,give_category=False,categories=[]):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(filename.split('.')[-2].split('/')[-1],f,give_category=give_category,categories=categories)
            except struct.error:
                break
            except json.decoder.JSONDecodeError:
                break

def interleave_generators(generators,dir_list=[],give_category=False,categories=[]):
    generators = [iter(gen) for gen in generators]
    nums = list(range(len(generators)))
    while len(nums) != 0:
        for gen in generators[:]:
            try:
                yield next(gen)
            except StopIteration:
                index = generators.index(gen)
                generators[index] = parse_data_from_file(f"./dataset/final/{dir_list[index]}",give_category=give_category,categories=categories)
                yield next(generators[index])
                try:
                    nums.remove(index)
                    print(nums)
                except:
                    pass

def parse_data():
    dir_list = os.listdir("./dataset/final")
    print(dir_list)
    length = len(dir_list)
    category = [x.split('.')[0] for x in dir_list]
    print(category)
    drawings = []
    for i in range(length):
        drawings.append(parse_data_from_file(f"./dataset/final/{dir_list[i]}"))
    drawings = interleave_generators(drawings,dir_list)
    print(type(next(drawings)))
    categories = []
    for i in range(length):
        categories.append(parse_data_from_file(f"./dataset/final/{dir_list[i]}",give_category=True,categories=category))
    categories = interleave_generators(categories,dir_list,give_category=True,categories=category)
    print(type(next(categories)))
    return drawings,categories

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(10, 20, 2)))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding="same", name="conv1d_1"))
    model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', padding="same", name="conv1d_2"))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same", name="conv1d_3"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(15,activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model


def train(data):

    pass

def test(data):
    pass

if __name__ == "__main__":
    model = create_model()
    x,y = parse_data()
    x = list(x)
    y = list(y)
    newx = []
    newy = []
    for i in range(len(x)):
        num_strokes = 9
        good = True
        for a in range(20):
            for b in range(2):
                if x[i][num_strokes][a][b] != 0:
                    good = False
        if good:
            --num_strokes
        for num in range(1,num_strokes+2):
            copy = np.copy(x[i])
            for a in range(num,10):
                for b in range(20):
                    for c in range(2):
                        copy[a][b][c] = 0
            newx.append(copy)
            newy.append(y[i])
    x = np.array(newx,dtype='float32')
    y = np.array(newy,dtype='float32')
    print("genereated data")
    print(x.shape)
    x_train = x[::2,:,:]
    x_verify = x[1::4,:,:]
    x_test = x[3::4,:,:]
    print(y.shape)
    y_train = y[::2,:]
    y_verify = y[1::4,:]
    y_test = y[3::4,:]
    print("split into test and training")
    model.fit(x_train,y_train,batch_size=256,epochs=30,validation_data=(x_verify,y_verify))
    print("model trained")
    model.evaluate(x_test,y_test)
    model.save("model11.keras")
    # questions = json.loads(open("test_data","r").read())
    # predict(questions)