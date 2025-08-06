import numpy as np
def ADB_solve(D):
    """
    Given a matrix D of size N x M, find all pairs of A, B such that A is a permutation, B is a rotation, and AD = DB

    equivalently, given a set of points, find all rotations that permute the points
    """

    N = len(D)
    eye = np.eye(N)

    iterations = [0]

    def test(D, prefix):
        iterations[0] += 1
        D = D - np.mean(D, axis=0, keepdims=True)
        if (
            np.abs(np.linalg.norm(D[len(prefix) - 1]) - np.linalg.norm(D[prefix[-1]]))
            > 0.001
        ):
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

        if np.abs(np.linalg.det(Q) + 1) < 0.001:
            return False
        if len(prefix) == N:
            permutations.append(A)
            rotations.append(Q)
            return False
        return (
            np.max(np.abs(transformed_A.T - Q @ transformed_identity.T)) < 0.01
        ).item()

    def recursive_solve(D, prefix):
        for i in range(N):
            oo = prefix + [i]
            if test(D, oo):
                recursive_solve(D, oo)

    permutations = []
    rotations = []
    print("solving")
    recursive_solve(D, [])
    print("solved")
    print(iterations)
    return permutations, rotations


