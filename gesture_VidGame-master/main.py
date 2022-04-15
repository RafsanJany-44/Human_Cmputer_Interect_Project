import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

import random
from time import sleep
import pygame

class CarRacing:
    def __init__(self):

        pygame.init()
        #pygame.camera.init()
        self.display_width = 800
        self.display_height = 600
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.clock = pygame.time.Clock()
        self.gameDisplay = None

        self.initialize()

    def initialize(self):

        self.crashed = False

        self.car_image = pygame.image.load('./img/car.png')
        self.car_x_coordinate = (self.display_width * 0.45)
        self.car_y_coordinate = (self.display_height * 0.8)
        self.car_width = 49

        # e_car
        self.e_car = pygame.image.load('./img/enemy_car_1.png')
        self.e_car_start_x = random.randrange(310, 450)
        self.e_car_start_y = -600
        self.e_car_velocity = 5
        self.e_car_width = 49
        self.e_car_height = 100
        # Background
        self.bgImg = pygame.image.load("./img/back_ground.jpg")
        self.background_x1 = (self.display_width / 2) - (360 / 2)
        self.background_x2 = (self.display_width / 2) - (360 / 2)
        self.background_y1 = 0
        self.background_y2 = -600
        self.background_velocity = 3
        self.count = 0

    def car(self, car_x_coordinate, car_y_coordinate):
        self.gameDisplay.blit(self.car_image, (car_x_coordinate, car_y_coordinate))

    def racing_window(self):
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Car Dodge')
        self.run_car()

    def run_car(self):

        while not self.crashed:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.crashed = True
                # print(event)

                if (event.type == pygame.KEYDOWN):
                    if (event.key == pygame.K_LEFT):
                        if (self.car_x_coordinate>=340):
                            self.car_x_coordinate -= 50
                        print ("CAR X COORDINATES: %s" % self.car_x_coordinate)
                    if (event.key == pygame.K_RIGHT):
                        if (self.car_x_coordinate < 440):
                            self.car_x_coordinate += 50
                        print ("CAR X COORDINATES: %s" % self.car_x_coordinate)
                    print ("x: {x}, y: {y}".format(x=self.car_x_coordinate, y=self.car_y_coordinate))

            self.gameDisplay.fill(self.black)
            self.back_ground_raod()

            self.run_enemy_car(self.e_car_start_x, self.e_car_start_y)
            self.e_car_start_y += self.e_car_velocity

            if self.e_car_start_y > self.display_height:
                self.e_car_start_y = 0 - self.e_car_height
                self.e_car_start_x = random.randrange(310, 450)

            self.car(self.car_x_coordinate, self.car_y_coordinate)
            self.highscore(self.count)
            self.count += 1
            if (self.count % 100 == 0):
                self.e_car_velocity += 1
                self.background_velocity += 1
            if self.car_y_coordinate < self.e_car_start_y + self.e_car_height:
                if self.car_x_coordinate > self.e_car_start_x and self.car_x_coordinate < self.e_car_start_x + self.e_car_width or self.car_x_coordinate + self.car_width > self.e_car_start_x and self.car_x_coordinate + self.car_width < self.e_car_start_x + self.e_car_width:
                    self.crashed = True
                    self.display_message("Game Over !!!")

            if self.car_x_coordinate < 310 or self.car_x_coordinate > 460:
                self.crashed = True
                self.display_message("Game Over !!!")

            pygame.display.update()
            self.clock.tick(60)

    def display_message(self, msg):
        font = pygame.font.SysFont("comicsansms", 72, True)
        text = font.render(msg, True, (255, 255, 255))
        self.gameDisplay.blit(text, (400 - text.get_width() // 2, 240 - text.get_height() // 2))
        self.display_credit()
        pygame.display.update()
        self.clock.tick(60)
        sleep(1)
        car_racing.initialize()
        car_racing.racing_window()

    def back_ground_raod(self):
        self.gameDisplay.blit(self.bgImg, (self.bg_x1, self.bg_y1))
        self.gameDisplay.blit(self.bgImg, (self.bg_x2, self.bg_y2))

        self.bg_y1 += self.background_velocity
        self.bg_y2 += self.background_velocity

        if self.bg_y1 >= self.display_height:
            self.bg_y1 = -600

        if self.bg_y2 >= self.display_height:
            self.bg_y2 = -600

    def run_enemy_car(self, thingx, thingy):
        self.gameDisplay.blit(self.e_car, (thingx, thingy))

    def highscore(self, count):
        font = pygame.font.SysFont("arial", 20)
        text = font.render("Score : " + str(count), True, self.white)
        self.gameDisplay.blit(text, (220, 0))

    def display_credit(self):
        font = pygame.font.SysFont("lucidaconsole", 14)
        text = font.render("Thanks for playing!", True, self.white)
        self.gameDisplay.blit(text, (600, 520))


car_racing = CarRacing()
car_racing.racing_window()
sleep(10)
