import numpy as np
import sys
import scipy
import asyncio


import pygame

pygame.init()


display_matrix = []
colors = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            count = 0
            color = []
            if i == 0:
                color = [255, 0, 0]
                count += 1
            if i == 4:
                color = [255, 255, 0]
                count += 1
            if j == 0:
                color = [0, 255, 0]
                count += 1
            if j == 4:
                color = [0, 255, 255]
                count += 1
            if k == 0:
                color = [0, 0, 255]
                count += 1
            if k == 4:
                color = [255, 0, 255]
                count += 1
                
            if count == 1:
                display_matrix.append([i, j, k])
                colors.append(color)
display_matrix = 200.5 + 60 * np.array(display_matrix)

font = pygame.font.SysFont("arial", 126)
fontB = pygame.font.SysFont("arial", 156)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



def solve(D):
    N = len(D)
    eye = np.eye(N)
    def test(D, prefix):
        D = D - np.mean(D, axis=0, keepdims=True)
        A = eye[prefix]
        targ = eye[: len(prefix)]
        transformed_A = A @ D
        transformed_identity = targ @ D
        Q = (
            np.linalg.inv(transformed_A.T @ transformed_A + np.eye(3) * 0.000001)
            @ transformed_A.T
            @ transformed_identity
        )
        
        if np.abs(np.linalg.det(Q) + 1) < .001:
            return False
        if len(prefix) == N + 3:
            if np.all(A @ A == eye):
                return False
            if np.all(A @ A @ A == eye):
                return False
        return (np.max(np.abs(transformed_A.T - Q @ transformed_identity.T)) < 0.001).item()
    def recursive_solve(D, prefix, out):
        if len(prefix) == N:
            out.append(eye[prefix])
            return
        for i in range(N):
            oo = prefix + [i]
            if test(D, oo):
                recursive_solve(D, oo, out)
    out = []
    recursive_solve(D, [], out)
    return out


rot = solve(display_matrix)

head = display_matrix[:21]

rot_face = solve(head)
template = np.eye(54)
template[:21, :21] = rot_face[1]

moves = [
]

def maybe_add(arr):
    if not any(np.all(arr == u) for u in moves):
        moves.append(arr)
for m in rot:
    maybe_add(m @ template @ m.T)
    maybe_add(m @ template.T @ m.T)
for m in rot[1:]:
    maybe_add(m)
    maybe_add(m.T)


frames_per_turn = 10

moves = [scipy.linalg.expm(1 / frames_per_turn * scipy.linalg.logm(x)) for x in moves]


def main():
    state = np.eye(54)
    toMove = []
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
                for i, letter in enumerate("qwerasdfzxcvtyuighjkbnm"):
                    if event.unicode == letter:
                        toMove += [i] * frames_per_turn
                if event.unicode == "p":
                    toMove = []
                    state = np.eye(54)
        if len(toMove):
            state = state @ moves[toMove[0]]
            toMove = toMove[1:]
        screen.fill(WHITE)
        
        cube = state.real @ display_matrix

        cube = reversed(sorted(list(zip(map(tuple, cube), map(list, colors)))))

        for i, ((z, x, y), color) in enumerate(cube):
            text_surface = fontB.render("●", True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.center = (x - .4 * z, y - .4 * z)

            screen.blit(text_surface, text_rect)
            text_surface = font.render("●", True, color)
            text_rect = text_surface.get_rect()
            text_rect.center = (x - .4 * z, y - .4 * z)

            screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


main()
