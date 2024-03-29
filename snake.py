from constants import *
import pygame as pg
class Snake():
    def __init__(self, surface):
        self.x = GAME_WIDTH // 2
        self.y = GAME_HEIGHT // 2
        self.surface = surface
        self.rect = pg.Rect(self.x, self.y, BLOCK_SIZE, BLOCK_SIZE)
        self.dir = (1, 0)
        self.score = 0



    def update(self):
        dir_x = self.dir[0] * BLOCK_SIZE
        dir_y = self.dir[1] * BLOCK_SIZE
        self.rect = self.rect.move(dir_x, dir_y)
        self.check_bounds()


    def check_bounds(self):
        if self.rect.x < 0 or self.rect.right > GAME_WIDTH or self.rect.y < 0 or self.rect.bottom > GAME_HEIGHT:
            self.restart()


    def restart(self):
        self.rect.x = GAME_WIDTH // 2
        self.rect.y = GAME_HEIGHT // 2
        self.score = 0
        self.dir = (1, 0)
        pg.time.delay(1000)



    def draw(self):
        pg.draw.rect(self.surface, (0, 215, 0), self.rect)


    def keyboard_controler(self, key):

        if key == pg.K_UP or key == pg.K_w:
            self.move_up()

        if key == pg.K_RIGHT or key == pg.K_d:
            self.move_right()

        if key == pg.K_DOWN or key == pg.K_s:
            self.move_down()

        if key == pg.K_LEFT or key == pg.K_a:
            self.move_left()



    def move_up(self):
        print("")
        self.dir = (0, -1)

    def move_right(self):
        print("right")
        self.dir = (1, 0)

    def move_down(self):
        print("down")
        self.dir = (0, 1)

    def move_left(self):
        print("left")
        self.dir = (-1, 0)



    def update_score(self):
        self.score += 1



    def get_coords(self):
        return (self.rect.x ,self.rect.y)
