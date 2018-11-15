import keras.models
import PIL
from PIL import ImageTk
from PIL import Image
import argparse
import pickle
import random
import sys
from tkinter import Tk, Label, Button, Canvas, PhotoImage

def predict():
    print("Predict")

def next():
    print("next")

def main(args):

    model = keras.models.load_model(args.model)
    #Chargement des données
    i=random.randint(0, 2001) - 1
    j=random.randint(0, 2001) - 1
    Xdata = pickle.load( open( "../Data/data_2000/X_train.p", "rb" ))
    left_data = []
    right_data = []
    n = len(Xdata)
    left_data.append(Xdata[i][0])
    right_data.append(Xdata[j][1])

    #Chargement des images
    my_array = Xdata[i][0].reshape((32, 32)).astype('uint8')
    im1 = Image.fromarray(my_array)
    im1.save("../Data/images/temp/im1.jpg")
    my_array = Xdata[i][1].reshape((32, 32)).astype('uint8')
    im2 = Image.fromarray(my_array)
    im2.save("../Data/images/temp/im2.jpg")

    #Construction de la fenetre
    fenetre = Tk()
    fenetre.geometry('500x500')

    #Ajout de l'image 1
    image1 = Image.open("../Data/images/temp/im1.jpg")
    photo1 = ImageTk.PhotoImage(image1)
    label1 = Label(image=photo1)
    label1.image = photo1

    #Ajout de l'image 2
    image2 = Image.open("../Data/images/temp/im2.jpg")
    photo2 = ImageTk.PhotoImage(image2)
    label2 = Label(image=photo2)
    label2.image = photo2

    #Ajout des images de validation
    photo1 = PhotoImage(file="../Data/true.png")
    canvasTrue = Canvas(fenetre,width=36, height=36)
    canvasTrue.create_image(2, 2, anchor='nw', image=photo1)

    photo2 = PhotoImage(file="../Data/false.png")
    canvasFalse = Canvas(fenetre,width=36, height=36)
    canvasFalse.create_image(2, 2, anchor='nw', image=photo2)

    val = 1
    if val == 1:
        canvasTrue.pack()
    else :
        canvasFalse.pack()

    #Ajout du bouton de prédiction
    predictButton = Button(fenetre, text="Pedict", command=predict)
    predictButton.pack()

    #Ajout du bouton suivant
    nextButton = Button(fenetre, text="Next", command=next)
    nextButton.pack()

    #Ajout du bouton exit
    exitButton=Button(fenetre, text="Exit", command=fenetre.quit)
    exitButton.pack()

    #Ajout des images à la fenetre
    label1.pack()
    label2.pack()

    #prédiction
    predictions = model.predict([left_data,right_data])
    print(predictions)

    fenetre.mainloop()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='File of the model', default='../Models/default.h5')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
