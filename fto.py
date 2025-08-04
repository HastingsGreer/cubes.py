import numpy as np
import random
import sys
import scipy
import asyncio


import pygame

pygame.init()


display_matrix = []
colors = []

import numpy as np

shape = []

for i in range(3):
  point = np.zeros(3)
  point[i] = 1
  shape.append(point)
  shape.append(-point)
shape = np.array(shape)
hull = scipy.spatial.ConvexHull(shape)

def subdivide_triangle_centers(A, B, C):
    """
    Subdivide triangle into 9 triangles and return their centers.

    Args:
        A, B, C: tuples (x, y) representing triangle vertices

    Returns:
        list of 9 tuples (x, y) representing triangle centers
    """
    # Generate subdivision points using barycentric coordinates
    points = {}
    for i in range(4):
        for j in range(4):
            if i + j <= 3:
                u, v, w = i/3, j/3, 1 - i/3 - j/3
                x = w*A[0] + u*B[0] + v*C[0]
                y = w*A[1] + u*B[1] + v*C[1]
                z = w*A[2] + u*B[2] + v*C[2]

                points[(i,j)] = (x, y, z)

    # Define 9 triangles by their grid indices
    triangles = [
        [(0,0), (1,0), (0,1)], [(1,0), (2,0), (1,1)], [(2,0), (3,0), (2,1)],
        [(1,0), (1,1), (0,1)], [(2,0), (2,1), (1,1)], [(0,1), (1,1), (0,2)],
        [(1,1), (2,1), (1,2)], [(1,1), (1,2), (0,2)], [(0,2), (1,2), (0,3)]
    ]

    # Calculate centers as centroids
    centers = []
    for tri in triangles:
        p1, p2, p3 = [np.array(points[idx]) for idx in tri]
        center = ((p1[0] + p2[0] + p3[0])/3, (p1[1] + p2[1] + p3[1])/3, (p1[2] + p2[2] + p3[2])/3)
        for p in (p1, p2, p3):
            centers.append((np.array(center) + 5 * p) /6)

    return centers

puzzle = []

pallete = []
for i in [60, 230]:
  for j in [60, 230]:
    for k in [60, 230]:
      pallete.append((i, j, k))
colors = []

for i, simplex in enumerate(hull.simplices):
  a = shape[simplex[0]]
  b = shape[simplex[1]]
  c = shape[simplex[2]]
  centers = subdivide_triangle_centers(a, b, c)
  for center in centers:
    puzzle.append(center)
    colors.append(pallete[i])

puzzle = np.array(puzzle)

display_matrix = 300.5 + 200 * puzzle


def sort_by_row_sum(A, B):
    """
    Sort matrices A and B by the row sums of A.

    Args:
        A: Nx3 numpy array
        B: Nx3 numpy array

    Returns:
        tuple: (A_sorted, B_sorted) both sorted by A's row sums
    """
    indices = np.argsort( 1 - (A.sum(axis=1) < 830), stable=True)

    print(sum(A.sum(axis=1) < 830))
    return A[indices], B[indices]

display_matrix, colors = sort_by_row_sum(display_matrix, np.array(colors))

font = pygame.font.SysFont("arial", 70)
fontB = pygame.font.SysFont("arial", 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



def solve(D):
    N = len(D)
    eye = np.eye(N)
    def test(D, prefix):
        D = D - np.mean(D, axis=0, keepdims=True)
        if np.abs(np.linalg.norm(D[len(prefix) - 1]) - np.linalg.norm(D[prefix[-1]])) > .001:
            return False
        if len(prefix) != len(set(prefix)):
            return False
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
        return (np.max(np.abs(transformed_A.T - Q @ transformed_identity.T)) < 0.01).item()
    def recursive_solve(D, prefix, out):
        if len(prefix) == N:
            out.append(eye[prefix])
            return
        for i in range(N):
            oo = prefix + [i]
            if test(D, oo):
                recursive_solve(D, oo, out)
    out = []
    print("solving")
    recursive_solve(D, [], out)
    print("solved")
    return out


rot = solve(display_matrix)
moves = [ ]
def maybe_add(arr):
    if not any(np.all(arr == u) for u in moves):
        moves.append(arr)

slice_size = 3 * (9 + 6 * 3)

print(slice_size)

head = display_matrix[:(slice_size)]
rot_face = solve(head)
template = np.eye(len(display_matrix))
template[:slice_size, :slice_size] = rot_face[1]

for m in rot:
    maybe_add(m @ template @ m.T)
    maybe_add(m @ template.T @ m.T)

for m in rot[1:]:
    if np.all(m @ m == np.eye(len(display_matrix))):
              continue
    if np.all(m @ m@m == np.eye(len(display_matrix))):
              continue
    maybe_add(m)
    maybe_add(m.T)
maybe_add(
    np.eye(len(display_matrix)))


frames_per_turn = 14

moves = [scipy.linalg.expm(1 / frames_per_turn * scipy.linalg.logm(x)) for x in moves]


np.random.seed(42); view = np.linalg.qr(np.random.randn(3,3))[0]

view = scipy.linalg.expm(1 / 10 * scipy.linalg.logm(view)).real
def main():
    state = np.eye(len(display_matrix))
    toMove = []
    

    for i in range(0):
        j = random.randint(0, len(moves) - 1)
        toMove += [j] * frames_per_turn

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
                for i, letter in enumerate("1234qwerasdfzxcv5678tyuighjkbnm,"):
                    if event.unicode == letter:
                        if i < len(moves):
                            toMove += [i] * frames_per_turn
                if event.unicode == "p":
                    toMove = []
                    toMove += [len(moves) - 1] * frames_per_turn
                    state = state.real
                    moves[-1] = scipy.linalg.expm(1 / frames_per_turn * scipy.linalg.logm(np.linalg.inv(state)))
        if len(toMove):
            state = state @ moves[toMove[0]]
            toMove = toMove[1:]
        screen.fill(WHITE)
        
        cube = state.real @ display_matrix @ view
        
        


        cube = list(zip(map(tuple, cube), map(list, colors)))

        
        triangle = []
        triangles = []
        for pt in cube:
            triangle.append( pt
                    )
            if len(triangle) == 3:
                triangles.append(triangle)
                triangle = []
        triangles = sorted(triangles)
        for triangle in triangles:
                ((z, x, y), color) = triangle[0]
                screen_space = [triangle[0][0][1:], triangle[1][0][1:], triangle[2][0][1:]]
                pygame.draw.polygon(screen, color, screen_space)
                pygame.draw.polygon(screen, BLACK, screen_space, width=6)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


main()
