from constants import *
import pygame
import snake
import apple

clock = pygame.time.Clock()
surface = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
game_speed = 0.5
pygame.init()


snake = snake.Snake(surface)
apple = apple.Apple(surface)
def main():
    running = True
    prev_time = pygame.time.get_ticks()

    while running:
        clock.tick(TICK_RATE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                snake.keyboard_controler(event.key)
                # keys = pygame.key.get_pressed()
                print(f"event.key:{event.key}")
                if event.key == pygame.K_r:
                    board.reset()
                if event.key == pygame.K_q:
                    running = False





        delta = pygame.time.get_ticks() - prev_time
        if delta > game_speed * 1000:
            prev_time = pygame.time.get_ticks()
            draw()
            update()

    pygame.quit()



def draw():
    surface.fill((0, 0, 0))#background
    snake.draw()
    apple.draw()
    pygame.display.flip()



def update():
    snake.update()
    apple.update(snake)


if __name__ == "__main__":
    main()
