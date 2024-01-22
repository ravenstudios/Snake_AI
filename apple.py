from constants import *
import pygame as pg
import random


class Apple():
    def __init__(self, surface):

        self.surface = surface
        self.rect = pg.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
        self.get_new_location()
        self.dir = (1, 0)
        self.score = 0



    def update(self, snake):
        coord = (self.rect.x, self.rect.y)
        sn = snake.get_coords()
        snake_coords = (sn[0], sn[1])
        if snake_coords == coord:
            snake.update_score()
            self.get_new_location()


    def draw(self):
        pg.draw.rect(self.surface, (215, 0, 0), self.rect)


    def get_new_location(self):
        x = random.randint(0, GRID_SIZE - 2)
        y = random.randint(0, GRID_SIZE - 2)
        self.rect.x, self.rect.y = x * BLOCK_SIZE, y * BLOCK_SIZE
        print(self.rect)

    def get_coords(self):
        return (self.rect.x ,self.rect.y)
