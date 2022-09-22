from cgitb import text
from logging import root
import sqlite3
from turtle import onrelease
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.app import App

from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivymd.uix.list import IRightBodyTouch,OneLineAvatarIconListItem
from kivymd.uix.behaviors import TouchBehavior
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.list import OneLineListItem

import cv2

from kivymd.app import MDApp

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense




class RightCheckbox(IRightBodyTouch, MDCheckbox):
     """Custom right container."""

class CameraPage(Image):

    def __init__(self, **kwargs):
        super(CameraPage, self).__init__(**kwargs)

        self.actions = np.array(['መስማት','መናገር','ማንበብ','ትምህርት ቤት','አማርኛ','አኔ'])

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.model.load_weights('new.h5')
        self.sentence = []
        self.sequence = []
        self.predictions = []
        self.threshold = 0.6
        mp_holistic = mp.solutions.holistic # Holistic model
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        conn = sqlite3.connect('detection.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE if not exists history(
            hist text
        )""")
        conn.commit()
        conn.close()
        
    def loadCamera(self,*args):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

    def CloseCamera(self,*args):
        self.capture.release()

    def update(self, *args):
        conn = sqlite3.connect('detection.db')
        c = conn.cursor()
        query  = ( """INSERT INTO history(hist) VALUES 
                          (?)""")
        ret, frame = self.capture.read()
        if ret:
            image, results = self.mediapipe_detection(frame, self.holistic)
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            sequence = self.sequence[-30:]
            if len(sequence) == 30:
                res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                self.predictions.append(np.argmax(res))

                if np.unique(self.predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > self.threshold:
                        print(self.actions[np.argmax(res)])
                        data = self.actions[np.argmax(res)]
                        c.execute('INSERT INTO history VALUES(?)', (data,))
                        conn.commit()
                        conn.close()
                        if len(self.sentence) > 0: 
                            if self.actions[np.argmax(res)] != self.sentence[-1]:
                                self.sentence.append(self.actions[np.argmax(res)])
                        else:
                            self.sentence.append(self.actions[np.argmax(res)])
                if len(self.sentence) > 3: 
                    self.sentence = self.sentence[-3:]
        word = ''
        for sent in self.sentence:
            word = f'{word}""{sent}'
            App.get_running_app().root.ids['pred'].text= f'{word}'
        if ret:
            bufImg = cv2.flip(frame, 0).tobytes()
            img_txtur = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_txtur.blit_buffer(bufImg, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = img_txtur

    def mediapipe_detection(self,image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            image.flags.writeable = False                  
            results = model.process(image)                 
            image.flags.writeable = True                   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
            return image, results

    def extract_keypoints(self,results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def history(self):
        conn = sqlite3.connect('detection.db')
        c = conn.cursor()
        c.execute( """SELECT DISTINCT hist FROM history""")
        records = c.fetchall()

        app = App.get_running_app()  # Return instance of the app
        for item in records:
             app.root.ids.scroll_list.add_widget(
                OneLineListItem(text=f"[font=Tera-Regular.ttf]{item[0]}[/font]" , on_release = self.detail)
            )

        conn.commit()
        conn.close()

    def detail(self, OneLineListItem):
        app = App.get_running_app()
        query  = OneLineListItem.text
        item = query.replace('[font=Tera-Regular.ttf]','')
        item1 = item.replace('[/font]','')
        if(item1== "መስማት"):
            app.root.ids.image.source = 'images/mesmat.gif'
        if(item1== "መናገር"):
            app.root.ids.image.source = 'images/menager.gif'
        if(item1== "ማንበብ"):
            app.root.ids.image.source = 'images/manbeb.gif'
        if(item1== "አኔ"):
            app.root.ids.image.source = 'images/ene.gif'
        if(item1== "ትምህርት ቤት"):
            app.root.ids.image.source = 'images/tmrtbet.gif'
        if(item1== "አለ"):
            app.root.ids.image.source = 'images/ale.gif'
        app.root.ids.nav_drawer.set_state("close")
        app.root.ids.screen_manager.current = "scr 5"
        
        


        
class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()


class Camera(MDApp):
    def build(self):
        return Builder.load_file('main.kv')

Camera().run()