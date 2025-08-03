import numpy as np
import scipy
import asyncio


import pygame

pygame.init()


display_matrix = []
for i in range(4):
    for j in range(4):
        display_matrix.append([j, i])
display_matrix = 30.5 + 60 * np.array(display_matrix)

A = np.array(
    [
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ]
)
font = pygame.font.Font(None, 32)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

N = 16

eye = np.eye(N)


def test(D, prefix):
    D = D - np.mean(D, axis=0, keepdims=True)
    A = eye[prefix]
    targ = eye[: len(prefix)]
    transformed_A = A @ D
    transformed_identity = targ @ D
    Q = (
        np.linalg.inv(transformed_A.T @ transformed_A + np.eye(2) * 0.000001)
        @ transformed_A.T
        @ transformed_identity
    )
    return (np.max(np.abs(transformed_A.T - Q @ transformed_identity.T)) < 0.001).item()


def recursive_solve(D, prefix, out):
    if len(prefix) == N:
        out.append(eye[prefix])
        return
    for i in range(16):
        oo = prefix + [i]
        if test(D, oo):
            recursive_solve(D, oo, out)


def solve(D):
    out = []
    recursive_solve(D, [], out)
    return out


rot = solve(display_matrix)[3]


moves = [
    A,
    rot @ A @ rot @ rot @ rot,
    rot @ rot @ A @ rot @ rot,
    rot @ rot @ rot @ A @ rot,
]

moves = [scipy.linalg.expm(0.1 * scipy.linalg.logm(x)) for x in moves]

flip = A @ A

toMove = [0, 0, 0, 0]


def main():
    state = np.eye(16)
    """
      Contains the game variables and loop
      """
    screen = pygame.display.set_mode((600, 500))
    clock = pygame.Clock()
    pygame.display.set_caption("Twiddle")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.unicode == "a":
                    toMove[0] += 10
                if event.unicode == "z":
                    toMove[1] += 10
                if event.unicode == "x":
                    toMove[2] += 10
                if event.unicode == "s":
                    toMove[3] += 10
        min_nonzero = 99
        index = 99
        for i in range(4):
            if toMove[i]:
                if toMove[i] < min_nonzero:
                    min_nonzero = toMove[i]
                    index = i

        if index != 99 and toMove[index]:
            toMove[index] -= 1
            state = state @ moves[index]
        screen.fill(WHITE)

        for i, (x, y) in enumerate(state.real @ display_matrix):
            text_surface = font.render(str(1 + i), True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.center = (x, y)

            screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


main()
